import * as cdk from 'aws-cdk-lib';
import * as cloudwatch from 'aws-cdk-lib/aws-cloudwatch';
import * as ec2 from 'aws-cdk-lib/aws-ec2';
import * as ecr from 'aws-cdk-lib/aws-ecr';
import * as ecs from 'aws-cdk-lib/aws-ecs';
import * as elbv2 from 'aws-cdk-lib/aws-elasticloadbalancingv2';
import * as iam from 'aws-cdk-lib/aws-iam';
import * as logs from 'aws-cdk-lib/aws-logs';
import * as secretsmanager from 'aws-cdk-lib/aws-secretsmanager';
import { Construct } from 'constructs';
import { VoiceEngineConfig, resourcePrefix } from '../config';

export interface EcsServiceSecretArns {
  readonly apiKey: string;
  readonly dailyApiKey: string;
  readonly assemblyAi: string;
  readonly elevenLabs: string;
}

export interface EcsServiceConstructProps {
  readonly config: VoiceEngineConfig;
  readonly vpc: ec2.IVpc;
  readonly taskSecurityGroup: ec2.ISecurityGroup;
  readonly privateSubnets: ec2.ISubnet[];
  /** ECR image tag to deploy. Read from `-c imageTag=...` context. */
  readonly imageTag: string;
  /** ECR repository hosting the image. */
  readonly repository: ecr.IRepository;
  readonly secretArns: EcsServiceSecretArns;
  readonly recordingsBucketArn: string;
  /** May be empty for prod until `.env.prod` is populated. */
  readonly recordingsKmsKeyArn: string;
  readonly targetGroup: elbv2.ApplicationTargetGroup;
  /**
   * Lambda name (or alias) the engine invokes for runtime-config and
   * call-record writes. Maps to `Settings.voice_api_lambda_name`.
   * Required by the engine's pydantic-settings boot — without it, the
   * task crashes on startup with ValidationError.
   */
  readonly voiceApiLambdaName: string;
}

/**
 * ECS Fargate cluster, task definition, service, and autoscaling.
 *
 * Why one construct (not three)
 * -----------------------------
 * Cluster / task definition / service / autoscaling are tightly coupled
 * — moving any one of them touches the others' permissions and naming.
 * Keeping them in one Construct lets the caller (ComputeStack) treat
 * "the engine fleet" as one logical unit and keeps the per-env divergence
 * (cooldowns, min/max capacity) local to one file.
 *
 * Task sizing — locked-in from Layer 9.5
 * --------------------------------------
 *   - 1 vCPU (1024 units), 2 GB memory (2048 MiB). Measured peak in
 *     scale test: 510 MB RSS, 66% CPU, file descriptors flat at ~215.
 *     2× headroom on memory for safety.
 *   - stopTimeout 120 s. Layer 9 graceful-drain budget is 90 s with
 *     30 s of slack for stragglers.
 *   - x86_64 / linux. The Dockerfile builds for amd64 explicitly.
 *
 * Auto-scaling — target tracking on `ActiveSessions`
 * --------------------------------------------------
 *   - Target metric: `VoiceAgent/Pipeline / ActiveSessions` (custom),
 *     dimension `{Environment}`, statistic Average, period 60s.
 *     Emitted by `app/runner/metrics.py` every 30s. Wave 3 task
 *     definition does NOT set ECS_TASK_ID, so the emitter publishes
 *     the Environment-only timeseries that this policy reads.
 *   - Target value: 70% of session capacity = floor(6 × 0.7) = 4.2.
 *     CDK accepts the fractional target; CloudWatch compares against it
 *     directly.
 *   - Scale-out cooldown 60s — capacity ramps fast under burst.
 *   - Scale-in cooldown 300s — slow drain prevents yo-yo on lulls.
 *
 * Deployment behaviour
 * --------------------
 *   - Rolling deploy with `minHealthyPercent=100, maxHealthyPercent=200`.
 *     Zero-downtime: a new task starts before any old task drains. With
 *     `desiredCount=1` (staging/prod start), the deploy temporarily
 *     runs 2 tasks then drains the old one.
 *   - Circuit breaker enabled with rollback. A failed deploy auto-rolls
 *     back rather than leaving the service in DRAINING limbo.
 *   - Health-check grace 30s — covers container cold-start (~10–15s
 *     observed locally with QEMU; native Fargate amd64 is faster).
 *   - ECS Exec enabled (`enableExecuteCommand: true`) for live debug.
 *     CDK auto-adds the ssmmessages IAM grants.
 *
 * IAM — least privilege per Layer 11 brief
 * ----------------------------------------
 *   - Execution role: ECR pull, CW Logs write, Secrets Manager
 *     GetSecretValue (for env-var injection of the 4 API keys).
 *   - Task role: bedrock invoke + invoke-stream, scoped Secrets Manager
 *     re-read (runtime SDK use), s3:PutObject on the recordings bucket,
 *     KMS encrypt/decrypt/generate-data-key on the recordings KMS key,
 *     lambda:InvokeFunction on `cosentus-voice-api-*`,
 *     cloudwatch:PutMetricData restricted to the
 *     `VoiceAgent/Pipeline` namespace, and ecs:UpdateTaskProtection
 *     scoped to this cluster's task ARNs.
 */
export class EcsServiceConstruct extends Construct {
  public readonly cluster: ecs.Cluster;
  public readonly taskDefinition: ecs.FargateTaskDefinition;
  public readonly service: ecs.FargateService;
  public readonly executionRole: iam.Role;
  public readonly taskRole: iam.Role;
  public readonly logGroup: logs.LogGroup;

  constructor(scope: Construct, id: string, props: EcsServiceConstructProps) {
    super(scope, id);
    const {
      config,
      vpc,
      taskSecurityGroup,
      privateSubnets,
      imageTag,
      repository,
      secretArns,
      recordingsBucketArn,
      recordingsKmsKeyArn,
      targetGroup,
      voiceApiLambdaName,
    } = props;

    const stack = cdk.Stack.of(this);
    const prefix = resourcePrefix(config);

    this.cluster = new ecs.Cluster(this, 'Cluster', {
      clusterName: `${prefix}-cluster`,
      vpc,
      containerInsightsV2: ecs.ContainerInsights.ENABLED,
    });

    this.logGroup = new logs.LogGroup(this, 'EngineLogGroup', {
      logGroupName: `/aws/ecs/${prefix}`,
      retention: logs.RetentionDays.ONE_MONTH,
      removalPolicy: cdk.RemovalPolicy.RETAIN,
    });

    this.executionRole = this.buildExecutionRole(prefix, secretArns);
    this.taskRole = this.buildTaskRole(
      prefix,
      stack,
      secretArns,
      recordingsBucketArn,
      recordingsKmsKeyArn,
      voiceApiLambdaName,
    );
    this.grantDailyRecordingUploaderRole(config, recordingsBucketArn);

    this.taskDefinition = new ecs.FargateTaskDefinition(this, 'TaskDefinition', {
      family: prefix,
      cpu: config.cpu,
      memoryLimitMiB: config.memoryMiB,
      executionRole: this.executionRole,
      taskRole: this.taskRole,
      runtimePlatform: {
        cpuArchitecture: ecs.CpuArchitecture.X86_64,
        operatingSystemFamily: ecs.OperatingSystemFamily.LINUX,
      },
    });

    // The api-key secret is fetched at runtime by the engine
    // (server.py::_load_api_key) via the API_KEY_SECRET_ARN env var.
    // No ecs.Secret injection here — the execution role's
    // secretsmanager:GetSecretValue grant covers the runtime fetch.
    const dailySecret = secretsmanager.Secret.fromSecretCompleteArn(
      this,
      'DailyApiKeySecretRef',
      secretArns.dailyApiKey,
    );
    const aaiSecret = secretsmanager.Secret.fromSecretCompleteArn(
      this,
      'AssemblyAiSecretRef',
      secretArns.assemblyAi,
    );
    const elevenSecret = secretsmanager.Secret.fromSecretCompleteArn(
      this,
      'ElevenLabsSecretRef',
      secretArns.elevenLabs,
    );
    const dailyRecordingWebhookHmacSecret = secretsmanager.Secret.fromSecretNameV2(
      this,
      'DailyRecordingWebhookHmacSecretRef',
      config.dailyRecordingWebhookHmacSecretName,
    );
    dailyRecordingWebhookHmacSecret.grantRead(this.executionRole);
    const recordingsBucketName = stack.splitArn(
      recordingsBucketArn,
      cdk.ArnFormat.NO_RESOURCE_NAME,
    ).resource;

    this.taskDefinition.addContainer('Engine', {
      containerName: 'engine',
      image: ecs.ContainerImage.fromEcrRepository(repository, imageTag),
      essential: true,
      logging: ecs.LogDrivers.awsLogs({
        streamPrefix: 'engine',
        logGroup: this.logGroup,
      }),
      portMappings: [
        {
          containerPort: 8080,
          protocol: ecs.Protocol.TCP,
        },
      ],
      stopTimeout: cdk.Duration.seconds(config.stopTimeoutSeconds),
      environment: {
        SERVICE_PORT: '8080',
        ENVIRONMENT: config.environment,
        AWS_REGION: config.region,
        MAX_CONCURRENT_CALLS: String(config.sessionCapacityPerTask),
        RECORDING_BUCKET: recordingsBucketName,
        RECORDING_ROLE_ARN: config.recordingRoleArn,
        RECORDING_REGION: config.recordingRegion,
        // Back-compat/diagnostic env var from prior infra. The engine reads
        // RECORDING_BUCKET through Settings.
        RECORDINGS_BUCKET_NAME: recordingsBucketName,
        // Required by Settings — engine refuses to boot without these.
        VOICE_API_LAMBDA_NAME: voiceApiLambdaName,
        API_KEY_SECRET_ARN: secretArns.apiKey,
        // v2 feature flags, per-environment (see config.ts `featureFlags`).
        // Spreads FLOWS_ENABLED / CONTEXT_SUMMARIZATION_ENABLED /
        // KNOWLEDGE_PREFETCH_ENABLED / IDENTITY_VERIFICATION_KEYS as task env
        // vars. TRACING_ENABLED is intentionally absent until Langfuse exists.
        ...config.featureFlags,
      },
      secrets: {
        // Vendor SDK keys read directly via os.environ in Layer 3
        // (factory.py) — tech debt entry 7 tracks that. The engine's
        // /start bearer auth uses Secrets Manager via api_key_secret_arn
        // (server.py::_load_api_key), so we DO NOT also inject API_KEY
        // here. apiKeySecret remains referenced so the IAM execution-role
        // policy grants GetSecretValue on the ARN.
        DAILY_API_KEY: ecs.Secret.fromSecretsManager(dailySecret),
        DAILY_RECORDING_WEBHOOK_HMAC: ecs.Secret.fromSecretsManager(
          dailyRecordingWebhookHmacSecret,
        ),
        ASSEMBLYAI_API_KEY: ecs.Secret.fromSecretsManager(aaiSecret),
        ELEVENLABS_API_KEY: ecs.Secret.fromSecretsManager(elevenSecret),
      },
    });

    this.service = new ecs.FargateService(this, 'Service', {
      serviceName: `${prefix}-service`,
      cluster: this.cluster,
      taskDefinition: this.taskDefinition,
      desiredCount: config.minCapacity,
      assignPublicIp: false,
      vpcSubnets: { subnets: privateSubnets },
      securityGroups: [taskSecurityGroup],
      healthCheckGracePeriod: cdk.Duration.seconds(30),
      minHealthyPercent: 100,
      maxHealthyPercent: 200,
      enableExecuteCommand: true,
      circuitBreaker: { rollback: true },
      propagateTags: ecs.PropagatedTagSource.SERVICE,
    });

    this.service.attachToApplicationTargetGroup(targetGroup);

    const scalableTarget = this.service.autoScaleTaskCount({
      minCapacity: config.minCapacity,
      maxCapacity: config.maxCapacity,
    });

    const targetValue =
      (config.sessionCapacityPerTask * config.targetSessionsPct) / 100;

    // Metric contract matches `app/runner/metrics.py`:
    //   namespace=VoiceAgent/Pipeline, name=ActiveSessions,
    //   dimensions={Environment}. Wave 3 task definition deliberately
    //   does NOT set ECS_TASK_ID, so the emitter publishes the
    //   Environment-only timeseries that this policy reads. If
    //   anything in the emitter changes (rename, new dimension),
    //   update here too.
    scalableTarget.scaleToTrackCustomMetric('ActiveSessionsTracking', {
      metric: new cloudwatch.Metric({
        namespace: 'VoiceAgent/Pipeline',
        metricName: 'ActiveSessions',
        dimensionsMap: { Environment: config.environment },
        statistic: 'Average',
        period: cdk.Duration.minutes(1),
      }),
      targetValue,
      scaleOutCooldown: cdk.Duration.seconds(config.scaleOutCooldownSeconds),
      scaleInCooldown: cdk.Duration.seconds(config.scaleInCooldownSeconds),
    });
  }

  private buildExecutionRole(prefix: string, secretArns: EcsServiceSecretArns): iam.Role {
    const role = new iam.Role(this, 'ExecutionRole', {
      roleName: `${prefix}-execution-role`,
      assumedBy: new iam.ServicePrincipal('ecs-tasks.amazonaws.com'),
      description: 'Pulls images, ships logs, injects secrets into the engine task at startup.',
    });
    role.addManagedPolicy(
      iam.ManagedPolicy.fromAwsManagedPolicyName('service-role/AmazonECSTaskExecutionRolePolicy'),
    );
    role.addToPolicy(
      new iam.PolicyStatement({
        actions: ['secretsmanager:GetSecretValue'],
        resources: Object.values(secretArns),
      }),
    );
    return role;
  }

  private buildTaskRole(
    prefix: string,
    stack: cdk.Stack,
    secretArns: EcsServiceSecretArns,
    recordingsBucketArn: string,
    recordingsKmsKeyArn: string,
    voiceApiLambdaName: string,
  ): iam.Role {
    const role = new iam.Role(this, 'TaskRole', {
      roleName: `${prefix}-task-role`,
      assumedBy: new iam.ServicePrincipal('ecs-tasks.amazonaws.com'),
      description: 'Engine application permissions: Bedrock, secrets, recordings, lambda, metrics, task protection.',
    });

    role.addToPolicy(
      new iam.PolicyStatement({
        sid: 'BedrockInvoke',
        actions: ['bedrock:InvokeModel', 'bedrock:InvokeModelWithResponseStream'],
        resources: ['*'],
      }),
    );

    role.addToPolicy(
      new iam.PolicyStatement({
        sid: 'SecretsManagerRuntime',
        actions: ['secretsmanager:GetSecretValue'],
        resources: Object.values(secretArns),
      }),
    );

    role.addToPolicy(
      new iam.PolicyStatement({
        sid: 'RecordingsBucketWrite',
        actions: ['s3:PutObject', 's3:PutObjectAcl', 's3:GetObject'],
        resources: [`${recordingsBucketArn}/*`],
      }),
    );

    if (recordingsKmsKeyArn) {
      role.addToPolicy(
        new iam.PolicyStatement({
          sid: 'RecordingsKmsEncryptDecrypt',
          actions: [
            'kms:Encrypt',
            'kms:Decrypt',
            'kms:GenerateDataKey',
            'kms:DescribeKey',
          ],
          resources: [recordingsKmsKeyArn],
        }),
      );
    }

    // Grant invoke on the configured voice-api Lambda. The
    // `voiceApiLambdaName` config value may include an `:alias`
    // suffix (e.g. `medcloud-voice-api:live`); strip it before
    // building the resource ARN so the grant matches both the
    // unqualified function and any of its aliases / versions.
    //
    // Trailing wildcard matches alias-qualified ARNs (Lambda
    // service formats those as `function:<name>:<alias>`), so
    // `function:medcloud-voice-api*` covers `:live`, `:$LATEST`,
    // and any future aliases. For the staging stub (whose name is
    // `cosentus-voice-api-staging-stub`) the same pattern works.
    //
    // Hardcoded `cosentus-voice-api-*` previously didn't match
    // prod's `medcloud-voice-api:live`, which 100%-broke calls in
    // prod with AccessDeniedException on agent-config load.
    // Caught 2026-05-28 during the prod PSTN smoke.
    const voiceApiFunctionName = voiceApiLambdaName.split(':')[0];
    role.addToPolicy(
      new iam.PolicyStatement({
        sid: 'VoiceApiLambdaInvoke',
        actions: ['lambda:InvokeFunction'],
        resources: [
          stack.formatArn({
            service: 'lambda',
            resource: 'function',
            resourceName: `${voiceApiFunctionName}*`,
            arnFormat: cdk.ArnFormat.COLON_RESOURCE_NAME,
          }),
        ],
      }),
    );

    role.addToPolicy(
      new iam.PolicyStatement({
        sid: 'CloudWatchMetricsScoped',
        actions: ['cloudwatch:PutMetricData'],
        resources: ['*'],
        conditions: {
          // Must match _METRIC_NAMESPACE in app/runner/metrics.py.
          // PutMetricData uses no resource ARN; namespace condition is
          // the only way to scope it.
          StringEquals: {
            'cloudwatch:namespace': 'VoiceAgent/Pipeline',
          },
        },
      }),
    );

    role.addToPolicy(
      new iam.PolicyStatement({
        sid: 'EcsTaskProtection',
        actions: ['ecs:UpdateTaskProtection', 'ecs:GetTaskProtection'],
        resources: [
          stack.formatArn({
            service: 'ecs',
            resource: 'task',
            resourceName: `${prefix}-cluster/*`,
            arnFormat: cdk.ArnFormat.SLASH_RESOURCE_NAME,
          }),
        ],
      }),
    );

    return role;
  }

  private grantDailyRecordingUploaderRole(
    config: VoiceEngineConfig,
    recordingsBucketArn: string,
  ): void {
    const role = iam.Role.fromRoleArn(
      this,
      'DailyRecordingsUploaderRole',
      config.recordingRoleArn,
      { mutable: true },
    );
    role.addToPrincipalPolicy(
      new iam.PolicyStatement({
        sid: 'DailyRecordingsBucketAccess',
        actions: ['s3:ListBucket', 's3:ListBucketMultipartUploads', 's3:ListBucketVersions'],
        resources: [recordingsBucketArn],
      }),
    );
    role.addToPrincipalPolicy(
      new iam.PolicyStatement({
        sid: 'DailyRecordingsObjectAccess',
        actions: [
          's3:PutObject',
          's3:GetObject',
          's3:GetObjectVersion',
          's3:AbortMultipartUpload',
          's3:ListMultipartUploadParts',
        ],
        resources: [`${recordingsBucketArn}/voice-recordings/*`],
      }),
    );
  }
}
