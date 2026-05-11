import * as cdk from 'aws-cdk-lib';
import * as ec2 from 'aws-cdk-lib/aws-ec2';
import { Construct } from 'constructs';
import { VoiceEngineConfig, resourcePrefix } from '../config';

export interface VpcConstructProps {
  readonly config: VoiceEngineConfig;
}

/**
 * VPC + networking primitives for the voice engine.
 *
 * Topology
 * --------
 *   - 3 AZs (config.maxAzs, defaults to 3).
 *   - Public subnets (one per AZ) for the ALB.
 *   - Private subnets with egress (one per AZ) for ECS tasks.
 *   - NAT Gateways: staging=1 (cost saving), prod=3 (full AZ HA).
 *
 * VPC endpoints (interface or gateway, depending on the service)
 * --------------------------------------------------------------
 * We add endpoints for services the engine hits constantly inside the
 * task. Each endpoint avoids a round-trip through the NAT gateway, which
 * saves both data-transfer cost and latency:
 *
 *   - S3 (gateway endpoint, free)         — recordings download via Daily,
 *                                            and any future direct S3 access.
 *   - Secrets Manager (interface)         — task pulls API keys at startup.
 *   - Bedrock Runtime (interface)         — every LLM call.
 *   - ECR API + ECR DKR (interface)       — Fargate pulls container images.
 *   - CloudWatch Logs (interface)         — structured log shipping.
 *   - CloudWatch Monitoring (interface)   — PutMetricData for ActiveSessions.
 *   - Lambda (interface)                  — engine → voice-api Lambda
 *                                            (cosentus-voice-api-lambda) call.
 *   - STS (interface)                     — IAM role-chaining for any
 *                                            assume-role flows.
 *
 * Bedrock and Lambda endpoints are the highest-volume ones; placing them
 * here is the cost lever that justifies the endpoint sprawl. Per AWS
 * pricing each interface endpoint is ~$7/mo per AZ × the AZ count, so 7
 * endpoints × 3 AZs = ~$147/mo for prod. NAT Gateway data-transfer cost
 * scales linearly with call volume; even modest Bedrock traffic pays for
 * the endpoints inside a month.
 *
 * Security groups
 * ---------------
 *   - `albSecurityGroup`: ingress 443 from 0.0.0.0/0 (internet-facing
 *     ALB serves Daily's webhook and the engine's /start endpoint).
 *   - `taskSecurityGroup`: ingress on the engine port (8080) only from
 *     the ALB security group; egress all (Fargate needs to reach Daily,
 *     Bedrock, ElevenLabs, AssemblyAI, Lambda).
 */
export class VpcConstruct extends Construct {
  public readonly vpc: ec2.Vpc;
  public readonly albSecurityGroup: ec2.SecurityGroup;
  public readonly taskSecurityGroup: ec2.SecurityGroup;

  constructor(scope: Construct, id: string, props: VpcConstructProps) {
    super(scope, id);
    const { config } = props;
    const prefix = resourcePrefix(config);

    this.vpc = new ec2.Vpc(this, 'Vpc', {
      vpcName: `${prefix}-vpc`,
      ipAddresses: ec2.IpAddresses.cidr(config.vpcCidr),
      maxAzs: config.maxAzs,
      natGateways: config.natGateways,
      subnetConfiguration: [
        {
          name: 'public',
          subnetType: ec2.SubnetType.PUBLIC,
          cidrMask: 24,
        },
        {
          name: 'private-with-egress',
          subnetType: ec2.SubnetType.PRIVATE_WITH_EGRESS,
          cidrMask: 22,
        },
      ],
      enableDnsHostnames: true,
      enableDnsSupport: true,
      restrictDefaultSecurityGroup: true,
    });

    this.albSecurityGroup = new ec2.SecurityGroup(this, 'AlbSecurityGroup', {
      vpc: this.vpc,
      securityGroupName: `${prefix}-alb-sg`,
      description: 'Voice engine ALB — accepts HTTPS from the public internet.',
      allowAllOutbound: true,
    });
    this.albSecurityGroup.addIngressRule(
      ec2.Peer.anyIpv4(),
      ec2.Port.tcp(443),
      'HTTPS from public internet (Daily webhooks + /start auth)',
    );

    this.taskSecurityGroup = new ec2.SecurityGroup(this, 'TaskSecurityGroup', {
      vpc: this.vpc,
      securityGroupName: `${prefix}-task-sg`,
      description: 'Voice engine Fargate tasks — accepts traffic from ALB only.',
      allowAllOutbound: true,
    });
    this.taskSecurityGroup.addIngressRule(
      this.albSecurityGroup,
      ec2.Port.tcp(8080),
      'Engine HTTP port — ALB only',
    );

    this.vpc.addGatewayEndpoint('S3GatewayEndpoint', {
      service: ec2.GatewayVpcEndpointAwsService.S3,
    });

    const interfaceEndpoints: Array<{ id: string; service: ec2.InterfaceVpcEndpointAwsService }> = [
      { id: 'SecretsManagerEndpoint', service: ec2.InterfaceVpcEndpointAwsService.SECRETS_MANAGER },
      { id: 'BedrockRuntimeEndpoint', service: ec2.InterfaceVpcEndpointAwsService.BEDROCK_RUNTIME },
      { id: 'EcrApiEndpoint', service: ec2.InterfaceVpcEndpointAwsService.ECR },
      { id: 'EcrDkrEndpoint', service: ec2.InterfaceVpcEndpointAwsService.ECR_DOCKER },
      { id: 'CloudWatchLogsEndpoint', service: ec2.InterfaceVpcEndpointAwsService.CLOUDWATCH_LOGS },
      { id: 'CloudWatchEndpoint', service: ec2.InterfaceVpcEndpointAwsService.CLOUDWATCH },
      { id: 'LambdaEndpoint', service: ec2.InterfaceVpcEndpointAwsService.LAMBDA },
      { id: 'StsEndpoint', service: ec2.InterfaceVpcEndpointAwsService.STS },
    ];

    for (const { id: endpointId, service } of interfaceEndpoints) {
      this.vpc.addInterfaceEndpoint(endpointId, {
        service,
        privateDnsEnabled: true,
        subnets: { subnetType: ec2.SubnetType.PRIVATE_WITH_EGRESS },
        securityGroups: [this.taskSecurityGroup],
      });
    }

    cdk.Tags.of(this.vpc).add('Component', 'network');
  }

  /** Subnet IDs the ECS service should run tasks in (private with egress). */
  public get taskSubnetIds(): string[] {
    return this.vpc.selectSubnets({ subnetType: ec2.SubnetType.PRIVATE_WITH_EGRESS }).subnetIds;
  }

  /** Subnet IDs the ALB should attach to (public). */
  public get albSubnetIds(): string[] {
    return this.vpc.selectSubnets({ subnetType: ec2.SubnetType.PUBLIC }).subnetIds;
  }
}
