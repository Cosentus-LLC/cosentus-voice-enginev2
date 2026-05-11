/**
 * SSM parameter keys for cross-stack communication.
 *
 * Pattern (lifted from v1)
 * ------------------------
 * Stacks publish their outputs as SSM parameters under a stable path, and
 * downstream stacks read them via `ssm.StringParameter.valueFromLookup`.
 * The CloudFormation cyclic-dependency story is broken by SSM acting as a
 * loose-coupled message bus.
 *
 * Naming convention
 * -----------------
 * `/cosentus-voice-engine/{environment}/{stack}/{output}`
 *
 * Stack names are kept lowercase + hyphenated; outputs are camelCase to
 * mirror the IaC-side property names.
 */

import { VoiceEngineConfig } from './config';

export function ssmParams(config: VoiceEngineConfig) {
  const base = `/cosentus-voice-engine/${config.environment}`;
  return {
    // EcrStack (env-independent, but key still scoped to env for symmetry)
    ECR_REPOSITORY_URI: `${base}/ecr/repositoryUri`,
    ECR_REPOSITORY_ARN: `${base}/ecr/repositoryArn`,

    // NetworkStack
    VPC_ID: `${base}/network/vpcId`,
    PRIVATE_SUBNET_IDS: `${base}/network/privateSubnetIds`,
    PUBLIC_SUBNET_IDS: `${base}/network/publicSubnetIds`,
    TASK_SECURITY_GROUP_ID: `${base}/network/taskSecurityGroupId`,
    ALB_SECURITY_GROUP_ID: `${base}/network/albSecurityGroupId`,

    // StorageStack
    API_KEY_SECRET_ARN: `${base}/storage/apiKeySecretArn`,
    DAILY_API_KEY_SECRET_ARN: `${base}/storage/dailyApiKeySecretArn`,
    ASSEMBLYAI_API_KEY_SECRET_ARN: `${base}/storage/assemblyAiApiKeySecretArn`,
    ELEVENLABS_API_KEY_SECRET_ARN: `${base}/storage/elevenLabsApiKeySecretArn`,
    RECORDINGS_BUCKET_NAME: `${base}/storage/recordingsBucketName`,
    RECORDINGS_BUCKET_ARN: `${base}/storage/recordingsBucketArn`,

    // CertStack (shared cert ARN — env-independent in practice, but per-env
    // SSM key keeps stack composition clean)
    CERTIFICATE_ARN: `${base}/cert/certificateArn`,

    // ComputeStack
    CLUSTER_ARN: `${base}/compute/clusterArn`,
    SERVICE_NAME: `${base}/compute/serviceName`,
    TASK_DEFINITION_ARN: `${base}/compute/taskDefinitionArn`,
    ALB_DNS_NAME: `${base}/compute/albDnsName`,
    ALB_HOSTED_ZONE_ID: `${base}/compute/albHostedZoneId`,
  } as const;
}

export type SsmParamKeys = ReturnType<typeof ssmParams>;
