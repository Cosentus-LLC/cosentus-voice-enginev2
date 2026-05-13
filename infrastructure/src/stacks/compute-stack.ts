import * as cdk from 'aws-cdk-lib';
import * as acm from 'aws-cdk-lib/aws-certificatemanager';
import * as ec2 from 'aws-cdk-lib/aws-ec2';
import * as ecr from 'aws-cdk-lib/aws-ecr';
import * as ssm from 'aws-cdk-lib/aws-ssm';
import { Construct } from 'constructs';
import { VoiceEngineConfig, resourcePrefix } from '../config';
import { SHARED_CERT_ARN_PARAM, ssmParams } from '../ssm-parameters';
import {
  AlbConstruct,
  EcsServiceConstruct,
  MonitoringConstruct,
} from '../constructs';

export interface ComputeStackProps extends cdk.StackProps {
  readonly config: VoiceEngineConfig;
  /** Container image tag to deploy. Resolved from CDK context `imageTag` or env var `IMAGE_TAG`; defaults to 'latest' for from-scratch synth. */
  readonly imageTag: string;
}

/**
 * Compute stack — ALB + ECS Fargate service + autoscaling + monitoring.
 *
 * All inputs come from SSM parameters published by NetworkStack +
 * StorageStack + EcrStack + CertStack. We use `valueForStringParameter`
 * (deploy-time CFN dynamic refs) rather than `valueFromLookup`
 * (synth-time AWS API calls), which means:
 *
 *   - First-time `cdk synth` works from scratch without prior network /
 *     storage / cert deploys. Synth produces CFN templates with
 *     `{{resolve:ssm:/...}}` placeholders that CloudFormation resolves
 *     at deploy.
 *   - Trade-off: deploys fail loudly if the SSM params don't exist yet.
 *     Real-world deploy order: ECR + Network + Storage + Cert first,
 *     then Compute.
 *
 * VPC reconstruction
 * ------------------
 * NetworkStack publishes subnet IDs as comma-separated SSM string lists.
 * For deploy-time resolution we read the SSM string, then `Fn::Split` it
 * and pick each AZ index via `Fn::Select`. The AZ count is fixed at
 * synth time by `config.maxAzs` (3) so we generate exactly that many
 * select-of-split tokens.
 *
 * AZ list
 * -------
 * `availabilityZones` is hardcoded to the us-east-1 trio
 * (`us-east-1a/b/c`) to avoid triggering a CDK environment lookup at
 * synth. The NetworkStack VPC is built with `maxAzs: 3` in the same
 * region, so the AZ list matches the actual subnet placement.
 */
export class ComputeStack extends cdk.Stack {
  public readonly alb: AlbConstruct;
  public readonly ecsService: EcsServiceConstruct;
  public readonly monitoring: MonitoringConstruct;

  constructor(scope: Construct, id: string, props: ComputeStackProps) {
    super(scope, id, props);
    const { config, imageTag } = props;
    const params = ssmParams(config);
    const prefix = resourcePrefix(config);

    const vpcId = ssm.StringParameter.valueForStringParameter(this, params.VPC_ID);
    const taskSgId = ssm.StringParameter.valueForStringParameter(
      this,
      params.TASK_SECURITY_GROUP_ID,
    );
    const albSgId = ssm.StringParameter.valueForStringParameter(
      this,
      params.ALB_SECURITY_GROUP_ID,
    );

    const privateSubnetIds = this.expandSubnetIds(
      params.PRIVATE_SUBNET_IDS,
      config.maxAzs,
      'private',
    );
    const publicSubnetIds = this.expandSubnetIds(
      params.PUBLIC_SUBNET_IDS,
      config.maxAzs,
      'public',
    );

    const apiKeySecretArn = ssm.StringParameter.valueForStringParameter(
      this,
      params.API_KEY_SECRET_ARN,
    );
    const dailyApiKeySecretArn = ssm.StringParameter.valueForStringParameter(
      this,
      params.DAILY_API_KEY_SECRET_ARN,
    );
    const assemblyAiSecretArn = ssm.StringParameter.valueForStringParameter(
      this,
      params.ASSEMBLYAI_API_KEY_SECRET_ARN,
    );
    const elevenLabsSecretArn = ssm.StringParameter.valueForStringParameter(
      this,
      params.ELEVENLABS_API_KEY_SECRET_ARN,
    );
    const recordingsBucketArn = ssm.StringParameter.valueForStringParameter(
      this,
      params.RECORDINGS_BUCKET_ARN,
    );

    const ecrRepositoryArn = ssm.StringParameter.valueForStringParameter(
      this,
      params.ECR_REPOSITORY_ARN,
    );

    const certificateArn = ssm.StringParameter.valueForStringParameter(
      this,
      SHARED_CERT_ARN_PARAM,
    );

    const vpc = ec2.Vpc.fromVpcAttributes(this, 'Vpc', {
      vpcId,
      availabilityZones: ['us-east-1a', 'us-east-1b', 'us-east-1c'],
      privateSubnetIds,
      publicSubnetIds,
    });

    const publicSubnets = publicSubnetIds.map((subnetId, idx) =>
      ec2.Subnet.fromSubnetId(this, `PublicSubnet${idx}`, subnetId),
    );
    const privateSubnets = privateSubnetIds.map((subnetId, idx) =>
      ec2.Subnet.fromSubnetId(this, `PrivateSubnet${idx}`, subnetId),
    );

    const taskSecurityGroup = ec2.SecurityGroup.fromSecurityGroupId(
      this,
      'TaskSecurityGroup',
      taskSgId,
      { mutable: false },
    );
    const albSecurityGroup = ec2.SecurityGroup.fromSecurityGroupId(
      this,
      'AlbSecurityGroup',
      albSgId,
      { mutable: false },
    );

    const certificate = acm.Certificate.fromCertificateArn(
      this,
      'WildcardCertificate',
      certificateArn,
    );

    const repository = ecr.Repository.fromRepositoryAttributes(this, 'EngineRepo', {
      repositoryArn: ecrRepositoryArn,
      repositoryName: config.projectName,
    });

    this.alb = new AlbConstruct(this, 'Alb', {
      config,
      vpc,
      albSecurityGroup,
      publicSubnets,
      certificate,
    });

    this.ecsService = new EcsServiceConstruct(this, 'EcsService', {
      config,
      vpc,
      taskSecurityGroup,
      privateSubnets,
      imageTag,
      repository,
      secretArns: {
        apiKey: apiKeySecretArn,
        dailyApiKey: dailyApiKeySecretArn,
        assemblyAi: assemblyAiSecretArn,
        elevenLabs: elevenLabsSecretArn,
      },
      recordingsBucketArn,
      recordingsKmsKeyArn: config.recordingsKmsKeyArn,
      targetGroup: this.alb.targetGroup,
      voiceApiLambdaName: config.voiceApiLambdaName,
    });

    this.monitoring = new MonitoringConstruct(this, 'Monitoring', {
      config,
      cluster: this.ecsService.cluster,
      service: this.ecsService.service,
      serviceName: `${prefix}-service`,
      targetGroup: this.alb.targetGroup,
      loadBalancer: this.alb.alb,
    });

    new ssm.StringParameter(this, 'AlbDnsNameParam', {
      parameterName: params.ALB_DNS_NAME,
      stringValue: this.alb.alb.loadBalancerDnsName,
      description: 'Public DNS name of the engine ALB; CNAME target for GoDaddy records.',
    });
    new ssm.StringParameter(this, 'AlbHostedZoneIdParam', {
      parameterName: params.ALB_HOSTED_ZONE_ID,
      stringValue: this.alb.alb.loadBalancerCanonicalHostedZoneId,
      description: 'ALB canonical hosted zone id.',
    });
    new ssm.StringParameter(this, 'ClusterArnParam', {
      parameterName: params.CLUSTER_ARN,
      stringValue: this.ecsService.cluster.clusterArn,
      description: 'ECS cluster ARN.',
    });
    new ssm.StringParameter(this, 'ServiceNameParam', {
      parameterName: params.SERVICE_NAME,
      stringValue: this.ecsService.service.serviceName,
      description: 'ECS service name.',
    });
    new ssm.StringParameter(this, 'TaskDefinitionArnParam', {
      parameterName: params.TASK_DEFINITION_ARN,
      stringValue: this.ecsService.taskDefinition.taskDefinitionArn,
      description: 'ECS task definition ARN (most recent revision at deploy time).',
    });

    new cdk.CfnOutput(this, 'AlbDnsName', {
      value: this.alb.alb.loadBalancerDnsName,
      description: `Add a CNAME at GoDaddy: ${config.serviceHostname} → this value.`,
    });
    new cdk.CfnOutput(this, 'ServiceHostname', { value: config.serviceHostname });
    new cdk.CfnOutput(this, 'ImageTag', { value: imageTag });
    new cdk.CfnOutput(this, 'AlarmTopicArn', {
      value: this.monitoring.alarmTopic.topicArn,
      description: 'Subscribe operator endpoints (email/Slack/PagerDuty) to this SNS topic via Console.',
    });
    new cdk.CfnOutput(this, 'DashboardUrl', {
      value: `https://${config.region}.console.aws.amazon.com/cloudwatch/home?region=${config.region}#dashboards:name=${prefix}-monitoring`,
    });
  }

  /**
   * Read a comma-separated SSM list parameter at deploy time and split
   * it into individual subnet IDs. The AZ count is known at synth time
   * (config.maxAzs), so we emit exactly that many `Fn::Select(N, Fn::Split(','))`
   * tokens. Each element is a Token that CloudFormation resolves at deploy.
   */
  private expandSubnetIds(parameterName: string, azCount: number, label: string): string[] {
    const joined = ssm.StringParameter.valueForStringParameter(this, parameterName);
    const split = cdk.Fn.split(',', joined, azCount);
    const out: string[] = [];
    for (let i = 0; i < azCount; i++) {
      out.push(cdk.Fn.select(i, split));
    }
    // Touch the label so it stays in the function signature for readability;
    // it's already implicit in the CDK construct IDs we use downstream.
    void label;
    return out;
  }
}
