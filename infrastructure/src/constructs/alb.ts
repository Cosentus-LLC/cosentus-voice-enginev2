import * as cdk from 'aws-cdk-lib';
import * as acm from 'aws-cdk-lib/aws-certificatemanager';
import * as ec2 from 'aws-cdk-lib/aws-ec2';
import * as elbv2 from 'aws-cdk-lib/aws-elasticloadbalancingv2';
import { Construct } from 'constructs';
import { VoiceEngineConfig, resourcePrefix } from '../config';

export interface AlbConstructProps {
  readonly config: VoiceEngineConfig;
  readonly vpc: ec2.IVpc;
  readonly albSecurityGroup: ec2.ISecurityGroup;
  readonly publicSubnets: ec2.ISubnet[];
  readonly certificate: acm.ICertificate;
}

/**
 * Internet-facing Application Load Balancer for the voice engine.
 *
 * Why ALB (not NLB) — note vs v1
 * ------------------------------
 * v1 ran an internal NLB on port 80 (no TLS), invoked by the bot-runner
 * Lambda only. v2 collapses that bot-runner and puts the engine's HTTP
 * surface (`/health`, `/ready`, `/start`, `/status`,
 * `/daily-dialin-webhook`) directly on the public ALB:
 *
 *   - Daily.co's webhook is a public HTTP POST and needs a public DNS
 *     name and a real cert; an internal NLB can't satisfy that.
 *   - HTTP-layer features matter: path-based routing (multiple endpoints
 *     on one listener), proper health checks, idle-timeout tuning.
 *
 * Listener layout
 * ---------------
 * One HTTPS listener on 443. No port-80 listener — there is no HTTP
 * fallback, the redirect we'd otherwise add would be a footgun if Daily
 * ever cached an `http://` URL.
 *
 *   default action  →  forward to the engine target group
 *
 * All five engine routes share the same target group; path-based
 * separation is handled inside the engine (Layer 9's aiohttp app),
 * not at the listener. Keeps the CFN footprint and the failure surface
 * smaller.
 *
 * Target group
 * ------------
 *   - Protocol HTTP on port 8080 (TLS is terminated at the ALB).
 *   - Target type IP (Fargate awsvpc networking).
 *   - Health check: GET /ready, expect 200, 15s interval, 5s timeout,
 *     2 healthy / 3 unhealthy to flip state. The /ready endpoint
 *     returns 503 when at capacity, which makes the ALB stop routing
 *     new connections to that task — exactly the behaviour we want
 *     during a soft-drain.
 *   - Deregistration delay: 120s, matches Fargate stopTimeout and the
 *     engine's drain budget (Layer 9.5).
 *
 * Security hardening
 * ------------------
 *   - `dropInvalidHeaderFields: true` — drops headers with non-ASCII /
 *     malformed names before they reach the engine.
 *   - TLS policy `RECOMMENDED_TLS` (TLS 1.2+ negotiated, weak ciphers off).
 *     Modern Daily clients negotiate TLS 1.3; older ones fall back to 1.2.
 *   - SG ingress is `0.0.0.0/0` on 443 only (set in `VpcConstruct`).
 */
export class AlbConstruct extends Construct {
  public readonly alb: elbv2.ApplicationLoadBalancer;
  public readonly targetGroup: elbv2.ApplicationTargetGroup;
  public readonly httpsListener: elbv2.ApplicationListener;

  constructor(scope: Construct, id: string, props: AlbConstructProps) {
    super(scope, id);
    const { config, vpc, albSecurityGroup, publicSubnets, certificate } = props;
    const prefix = resourcePrefix(config);

    // ALB and TG names are 1-32 chars each. `cosentus-voice-engine-staging`
    // is 29; with a `-alb` suffix it becomes 33 and AWS rejects it. We use
    // `-lb` (32 char total for staging, 29 for prod) so both envs fit.
    this.alb = new elbv2.ApplicationLoadBalancer(this, 'Alb', {
      loadBalancerName: `${prefix}-lb`,
      vpc,
      internetFacing: true,
      securityGroup: albSecurityGroup,
      vpcSubnets: { subnets: publicSubnets },
      idleTimeout: cdk.Duration.seconds(60),
      dropInvalidHeaderFields: true,
      http2Enabled: true,
    });

    this.targetGroup = new elbv2.ApplicationTargetGroup(this, 'EngineTargetGroup', {
      targetGroupName: `${prefix}-tg`,
      vpc,
      port: 8080,
      protocol: elbv2.ApplicationProtocol.HTTP,
      targetType: elbv2.TargetType.IP,
      deregistrationDelay: cdk.Duration.seconds(120),
      // Health-check tolerances tuned by Wave 6 Option I (2026-05-18).
      // Previous values (interval=15, timeout=5, unhealthyThreshold=3)
      // were tripped by transient CPU spikes during mid-range sustained
      // traffic: a saturated task's /ready replies slower than 5 s, ALB
      // marked it unhealthy after 3 × 15 s = 45 s, ECS replaced it, and
      // the ~60 s replacement gap left no healthy targets. Loosening to
      // (30, 10, 4) gives transient saturation time to recover before
      // we tear down a still-functional task. The trade-off is slower
      // detection of a TRULY-broken task: ~120 s vs the previous ~45 s.
      // Autoscaler scale-out (target=40 % per task) now reacts before
      // we get near these thresholds in practice. See tech-debt entry 16.
      healthCheck: {
        path: '/ready',
        protocol: elbv2.Protocol.HTTP,
        port: '8080',
        healthyHttpCodes: '200',
        interval: cdk.Duration.seconds(30),
        timeout: cdk.Duration.seconds(10),
        healthyThresholdCount: 2,
        unhealthyThresholdCount: 4,
      },
    });

    this.httpsListener = this.alb.addListener('HttpsListener', {
      port: 443,
      protocol: elbv2.ApplicationProtocol.HTTPS,
      certificates: [certificate],
      sslPolicy: elbv2.SslPolicy.RECOMMENDED_TLS,
      defaultAction: elbv2.ListenerAction.forward([this.targetGroup]),
    });
  }
}
