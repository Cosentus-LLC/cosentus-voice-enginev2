/**
 * Per-environment CDK configuration.
 *
 * Source precedence (highest wins):
 *   1. CDK context (`cdk synth -c environment=prod`)
 *   2. Environment variables (`ENVIRONMENT=prod npm run synth:prod`)
 *   3. `.env.{environment}` file (loaded via dotenv)
 *   4. Built-in per-env defaults below
 *
 * All values are validated at synth time. Validation failures abort synth
 * loudly — better than discovering a typo in production CloudFormation.
 *
 * Locked-in choices baked into the defaults
 * -----------------------------------------
 * - Region us-east-1. v1 ships here, Bedrock Sonnet/Haiku inference profiles
 *   live here, and the Daily PSTN trunk routes through here. Multi-region is
 *   out of scope for v2.
 * - Per-task sizing: 1 vCPU, 2 GB memory, stopTimeout=120s, max_concurrent
 *   _calls=6. Validated empirically in Layer 9.5 — RSS 510 MB, CPU 66% peak,
 *   FDs flat at ~215. See docs/v2-tech-debt-log.md entry 13 for the burst
 *   load story.
 * - NAT gateway counts: staging=1 (~$35/mo saved vs full HA, acceptable for
 *   non-customer-facing env), prod=3 (one per AZ, full AZ-failure tolerance).
 */

import * as path from 'path';
import * as dotenv from 'dotenv';
import { App } from 'aws-cdk-lib';

export type Environment = 'staging' | 'prod';

export interface VoiceEngineConfig {
  /** Deployment environment — staging or prod. */
  readonly environment: Environment;
  /** AWS account; defaults to the medcloud-voice account (825269749545). */
  readonly account: string;
  /** AWS region; pinned to us-east-1 for v2. */
  readonly region: string;
  /** Project name prefix for all resource names. */
  readonly projectName: string;
  /** VPC CIDR — non-overlapping with v1's vpc-05a1f6c68c04943ec. */
  readonly vpcCidr: string;
  /** AZ count for the VPC (3 = full HA on Fargate). */
  readonly maxAzs: number;
  /** NAT gateway count. Staging=1, prod=3. */
  readonly natGateways: number;
  /** Fargate task definition vCPU. Locked at 1 (1024 units). */
  readonly cpu: number;
  /** Fargate task definition memory in MiB. Locked at 2048. */
  readonly memoryMiB: number;
  /**
   * Auto-scaling min task count.
   *
   * Staging=2 and prod=2 (Wave 6 Option I, 2026-05-18). Single-task
   * fleet is one ECS health-check timeout away from total outage —
   * we saw exactly this during Phase A revalidation when sustained
   * mid-range traffic pushed a single task into CPU saturation and
   * triggered a health-check-driven replacement. Cost of min=2 over
   * min=1: ~$30/mo Fargate. Cost of single-task outage: a call
   * volume's worth of dropped traffic.
   */
  readonly minCapacity: number;
  /**
   * Auto-scaling max task count.
   *
   * Staging=5 (covers validation N=6 scenario at one-call-per-task).
   * Prod=25 (headroom for 6 concurrent × 25 tasks = 150 concurrent calls).
   */
  readonly maxCapacity: number;
  /**
   * Target percent of session capacity that drives scale-out.
   *
   * 40 = scale out at 40% (= 2.4 sessions/task). Wave 6 Option I
   * (2026-05-18) lowered this from 70 (= 4.2/task) because target=4.2
   * gave the autoscaler too little headroom — sustained mid-range
   * traffic saturated CPU on the still-single task before scale-out
   * had time to react. 40 gives the autoscaler a generous head start.
   * Trade-off: more frequent scaling events under mixed traffic.
   */
  readonly targetSessionsPct: number;
  /** Scale-out cooldown — fast (60s) so capacity ramps quickly under burst. */
  readonly scaleOutCooldownSeconds: number;
  /** Scale-in cooldown — slow (300s) so we don't yo-yo on transient lulls. */
  readonly scaleInCooldownSeconds: number;
  /** Max concurrent calls per task (must match engine's max_concurrent_calls). */
  readonly sessionCapacityPerTask: number;
  /** ECS task stopTimeout in seconds. Locked at 120 (matches Layer 9 drain budget). */
  readonly stopTimeoutSeconds: number;
  /** Apex domain managed at the registrar (GoDaddy). */
  readonly domainApex: string;
  /** Per-env hostname pointing at the ALB (e.g., staging.cosentusaibackend.com). */
  readonly serviceHostname: string;
  /** Shared wildcard ACM cert ARN. Empty string until CertStack is deployed. */
  readonly certificateArn: string;
  /**
   * cosentus-voice-api Lambda function name (or alias) the engine
   * invokes for runtime-config + call-record writes. Required by the
   * engine's pydantic-settings boot. Default `medcloud-voice-api:live`
   * matches the v1 production Lambda. Staging deploys SHOULD repoint
   * at a staging alias before traffic flows — using prod here is a
   * deploy-validation-only shortcut.
   */
  readonly voiceApiLambdaName: string;
  /** Daily.co S3 recordings bucket name (existing or to-be-created). */
  readonly recordingsBucketName: string;
  /** IAM role Daily assumes to write recordings into the recordings bucket. */
  readonly recordingRoleArn: string;
  /** Region of the recordings bucket passed to Daily room config. */
  readonly recordingRegion: string;
  /** Existing Secrets Manager secret name for Daily recording webhook HMAC verification. */
  readonly dailyRecordingWebhookHmacSecretName: string;
  /** True when CDK owns the bucket lifecycle; false when we import an existing bucket. */
  readonly recordingsBucketOwnedByCdk: boolean;
  /**
   * KMS key ARN protecting the recordings bucket.
   *
   * - Staging: empty here, set at synth time by `RecordingsBucketConstruct`
   *   from the CDK-created key. The empty default is intentional — we don't
   *   know the key ARN until CloudFormation runs.
   * - Prod: must be populated via `.env.prod` (`RECORDINGS_KMS_KEY_ARN`)
   *   with the ARN of the existing v1 key that encrypts
   *   `medcloud-voice-us-prod-825`. If empty for prod, ComputeStack will
   *   still synth but the resulting task-role policy will not grant decrypt
   *   on a real key — fail-fast at deploy time, not at synth time.
   */
  readonly recordingsKmsKeyArn: string;
  /**
   * v2 feature-flag env vars set on the engine task definition. Each key maps
   * to a pydantic `Settings` flag; an absent flag means off. `TRACING_ENABLED`
   * is intentionally excluded until self-hosted Langfuse exists — otherwise the
   * engine would export OTLP spans to a dead endpoint.
   */
  readonly featureFlags: Record<string, string>;
}

const PROJECT_NAME = 'cosentus-voice-engine';
const DEFAULT_ACCOUNT = '825269749545';
const DEFAULT_REGION = 'us-east-1';
const DAILY_RECORDINGS_UPLOADER_ROLE_ARN =
  'arn:aws:iam::825269749545:role/daily-recordings-uploader';
const DAILY_RECORDING_WEBHOOK_HMAC_SECRET_NAME = 'voice-agent/daily-recording-webhook-hmac';

// The modernized v2 behavior — ON in both staging and prod: Pipecat Flows,
// the identity gate, payer-knowledge prefetch, and context summarization.
// TRACING_ENABLED is deliberately omitted until Langfuse is stood up.
const V2_FEATURE_FLAGS: Record<string, string> = {
  FLOWS_ENABLED: 'true',
  CONTEXT_SUMMARIZATION_ENABLED: 'true',
  KNOWLEDGE_PREFETCH_ENABLED: 'true',
  IDENTITY_VERIFICATION_KEYS: 'last_name',
};

const ENV_DEFAULTS: Record<Environment, Partial<VoiceEngineConfig>> = {
  staging: {
    vpcCidr: '10.20.0.0/16',
    maxAzs: 3,
    natGateways: 1,
    cpu: 1024,
    memoryMiB: 2048,
    // Wave 6 Option I (2026-05-18). minCapacity bumped from 1 to 2:
    // single-task fleet is one ECS health-check timeout away from
    // total outage. Cost: ~$30/mo extra Fargate. Worth the resilience.
    minCapacity: 2,
    maxCapacity: 5,
    // Wave 6 Option I (2026-05-18). targetSessionsPct lowered from 70
    // (= 4.2 sessions/task) to 40 (= 2.4 sessions/task) so the
    // autoscaler scales out BEFORE the single-task CPU bottleneck
    // and ALB-health-check timeout cascade observed in Phase A
    // revalidation. Trade-off: more frequent scaling events under
    // mixed traffic, but each call's CPU headroom is dramatically
    // bigger. See tech-debt entry 16 for the failure mode.
    targetSessionsPct: 40,
    scaleOutCooldownSeconds: 60,
    scaleInCooldownSeconds: 300,
    sessionCapacityPerTask: 6,
    stopTimeoutSeconds: 120,
    serviceHostname: 'staging.cosentusaibackend.com',
    recordingsBucketName: 'cosentus-voice-recordings-staging',
    recordingRoleArn: DAILY_RECORDINGS_UPLOADER_ROLE_ARN,
    recordingRegion: DEFAULT_REGION,
    dailyRecordingWebhookHmacSecretName: DAILY_RECORDING_WEBHOOK_HMAC_SECRET_NAME,
    recordingsBucketOwnedByCdk: true,
    recordingsKmsKeyArn: '',
    // Staging points at the stub Lambda (see VoiceApiStubStack) so
    // call-record writes don't touch production Aurora. The stub
    // returns canned runtime-config + accepts writes as no-ops.
    // Repoint at a real staging Lambda alias in Phase 6.
    voiceApiLambdaName: 'cosentus-voice-api-staging-stub',
    featureFlags: V2_FEATURE_FLAGS,
  },
  prod: {
    vpcCidr: '10.30.0.0/16',
    maxAzs: 3,
    natGateways: 3,
    cpu: 1024,
    memoryMiB: 2048,
    // Wave 6 Option I (2026-05-18). minCapacity bumped from 1 to 2
    // matching staging. Single-task fleet is one ECS health-check
    // timeout away from total outage; prod can't accept that risk.
    // Cost: ~$30/mo extra Fargate over the prior min=1 plan.
    minCapacity: 2,
    maxCapacity: 25,
    // Wave 6 Option I (2026-05-18). targetSessionsPct lowered from 70
    // to 40 (= 2.4 sessions/task) matching staging. Scales out
    // before single-task CPU saturation. See tech-debt entry 16.
    targetSessionsPct: 40,
    scaleOutCooldownSeconds: 60,
    scaleInCooldownSeconds: 300,
    sessionCapacityPerTask: 6,
    stopTimeoutSeconds: 120,
    serviceHostname: 'api.cosentusaibackend.com',
    recordingsBucketName: 'medcloud-voice-us-prod-825',
    recordingRoleArn: DAILY_RECORDINGS_UPLOADER_ROLE_ARN,
    recordingRegion: DEFAULT_REGION,
    dailyRecordingWebhookHmacSecretName: DAILY_RECORDING_WEBHOOK_HMAC_SECRET_NAME,
    recordingsBucketOwnedByCdk: false,
    recordingsKmsKeyArn: '',
    voiceApiLambdaName: 'medcloud-voice-api:live',
    featureFlags: V2_FEATURE_FLAGS,
  },
};

function readSetting(app: App, contextKey: string, envKey: string): string | undefined {
  const fromContext = app.node.tryGetContext(contextKey);
  if (typeof fromContext === 'string' && fromContext.length > 0) return fromContext;
  const fromEnv = process.env[envKey];
  if (typeof fromEnv === 'string' && fromEnv.length > 0) return fromEnv;
  return undefined;
}

function requireEnvironment(raw: string | undefined): Environment {
  if (raw !== 'staging' && raw !== 'prod') {
    throw new Error(
      `Invalid environment '${raw ?? '(unset)'}'. Set ENVIRONMENT or pass ` +
        `'-c environment=staging|prod'.`,
    );
  }
  return raw;
}

function requireCidr(raw: string): string {
  if (!/^(\d{1,3}\.){3}\d{1,3}\/\d{1,2}$/.test(raw)) {
    throw new Error(`Invalid CIDR '${raw}'.`);
  }
  return raw;
}

function requirePositiveInt(raw: string, label: string, min = 1, max = 100): number {
  const parsed = parseInt(raw, 10);
  if (Number.isNaN(parsed) || parsed < min || parsed > max) {
    throw new Error(`Invalid ${label} '${raw}'. Must be integer in [${min}, ${max}].`);
  }
  return parsed;
}

/**
 * Load + validate config for the resolved environment. Reads from CDK
 * context, env vars, and `.env.{environment}` files in that order.
 */
export function loadConfig(app: App): VoiceEngineConfig {
  const environment = requireEnvironment(
    readSetting(app, 'environment', 'ENVIRONMENT'),
  );

  dotenv.config({
    path: path.resolve(__dirname, '..', `.env.${environment}`),
  });

  const defaults = ENV_DEFAULTS[environment];
  const account =
    readSetting(app, 'account', 'CDK_DEFAULT_ACCOUNT') ??
    process.env.AWS_ACCOUNT_ID ??
    DEFAULT_ACCOUNT;
  const region =
    readSetting(app, 'region', 'CDK_DEFAULT_REGION') ?? DEFAULT_REGION;

  if (region !== DEFAULT_REGION) {
    throw new Error(
      `v2 is pinned to ${DEFAULT_REGION}. Refusing to synth for ${region}.`,
    );
  }

  const vpcCidr = requireCidr(
    readSetting(app, 'vpcCidr', 'VPC_CIDR') ?? (defaults.vpcCidr as string),
  );
  const maxAzs = requirePositiveInt(
    readSetting(app, 'maxAzs', 'MAX_AZS') ?? String(defaults.maxAzs),
    'maxAzs',
    1,
    6,
  );
  const natGateways = requirePositiveInt(
    readSetting(app, 'natGateways', 'NAT_GATEWAYS') ?? String(defaults.natGateways),
    'natGateways',
    0,
    maxAzs,
  );
  const cpu = requirePositiveInt(
    readSetting(app, 'cpu', 'CPU') ?? String(defaults.cpu),
    'cpu',
    256,
    16384,
  );
  const memoryMiB = requirePositiveInt(
    readSetting(app, 'memoryMiB', 'MEMORY_MIB') ?? String(defaults.memoryMiB),
    'memoryMiB',
    512,
    65536,
  );
  const minCapacity = requirePositiveInt(
    readSetting(app, 'minCapacity', 'MIN_CAPACITY') ?? String(defaults.minCapacity),
    'minCapacity',
    1,
    100,
  );
  const maxCapacity = requirePositiveInt(
    readSetting(app, 'maxCapacity', 'MAX_CAPACITY') ?? String(defaults.maxCapacity),
    'maxCapacity',
    minCapacity,
    200,
  );
  const targetSessionsPct = requirePositiveInt(
    readSetting(app, 'targetSessionsPct', 'TARGET_SESSIONS_PCT') ??
      String(defaults.targetSessionsPct),
    'targetSessionsPct',
    10,
    95,
  );
  const scaleOutCooldownSeconds = requirePositiveInt(
    readSetting(app, 'scaleOutCooldownSeconds', 'SCALE_OUT_COOLDOWN_SECONDS') ??
      String(defaults.scaleOutCooldownSeconds),
    'scaleOutCooldownSeconds',
    30,
    900,
  );
  const scaleInCooldownSeconds = requirePositiveInt(
    readSetting(app, 'scaleInCooldownSeconds', 'SCALE_IN_COOLDOWN_SECONDS') ??
      String(defaults.scaleInCooldownSeconds),
    'scaleInCooldownSeconds',
    30,
    3600,
  );
  const sessionCapacityPerTask = requirePositiveInt(
    readSetting(app, 'sessionCapacityPerTask', 'SESSION_CAPACITY_PER_TASK') ??
      String(defaults.sessionCapacityPerTask),
    'sessionCapacityPerTask',
    1,
    50,
  );
  const stopTimeoutSeconds = requirePositiveInt(
    readSetting(app, 'stopTimeoutSeconds', 'STOP_TIMEOUT_SECONDS') ??
      String(defaults.stopTimeoutSeconds),
    'stopTimeoutSeconds',
    30,
    120,
  );

  const domainApex =
    readSetting(app, 'domainApex', 'DOMAIN_APEX') ?? 'cosentusaibackend.com';
  const serviceHostname =
    readSetting(app, 'serviceHostname', 'SERVICE_HOSTNAME') ??
    (defaults.serviceHostname as string);
  const certificateArn = readSetting(app, 'certificateArn', 'CERTIFICATE_ARN') ?? '';

  const recordingsBucketName =
    readSetting(app, 'recordingsBucketName', 'RECORDINGS_BUCKET_NAME') ??
    (defaults.recordingsBucketName as string);
  const recordingRoleArn =
    readSetting(app, 'recordingRoleArn', 'RECORDING_ROLE_ARN') ??
    (defaults.recordingRoleArn as string);
  const recordingRegion =
    readSetting(app, 'recordingRegion', 'RECORDING_REGION') ??
    (defaults.recordingRegion as string);
  const dailyRecordingWebhookHmacSecretName =
    readSetting(
      app,
      'dailyRecordingWebhookHmacSecretName',
      'DAILY_RECORDING_WEBHOOK_HMAC_SECRET_NAME',
    ) ?? (defaults.dailyRecordingWebhookHmacSecretName as string);
  const recordingsBucketOwnedByCdk =
    (readSetting(app, 'recordingsBucketOwnedByCdk', 'RECORDINGS_BUCKET_OWNED_BY_CDK') ??
      String(defaults.recordingsBucketOwnedByCdk)) === 'true';
  const recordingsKmsKeyArn =
    readSetting(app, 'recordingsKmsKeyArn', 'RECORDINGS_KMS_KEY_ARN') ??
    (defaults.recordingsKmsKeyArn as string);
  const voiceApiLambdaName =
    readSetting(app, 'voiceApiLambdaName', 'VOICE_API_LAMBDA_NAME') ??
    (defaults.voiceApiLambdaName as string);

  return {
    environment,
    account,
    region,
    projectName: PROJECT_NAME,
    vpcCidr,
    maxAzs,
    natGateways,
    cpu,
    memoryMiB,
    minCapacity,
    maxCapacity,
    targetSessionsPct,
    scaleOutCooldownSeconds,
    scaleInCooldownSeconds,
    sessionCapacityPerTask,
    stopTimeoutSeconds,
    domainApex,
    serviceHostname,
    certificateArn,
    recordingsBucketName,
    recordingRoleArn,
    recordingRegion,
    dailyRecordingWebhookHmacSecretName,
    recordingsBucketOwnedByCdk,
    recordingsKmsKeyArn,
    voiceApiLambdaName,
    featureFlags: defaults.featureFlags ?? {},
  };
}

/** Resource prefix used by every stack and construct. */
export function resourcePrefix(config: VoiceEngineConfig): string {
  return `${config.projectName}-${config.environment}`;
}
