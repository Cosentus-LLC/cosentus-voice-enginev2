import * as cdk from 'aws-cdk-lib';
import * as ssm from 'aws-cdk-lib/aws-ssm';
import { Construct } from 'constructs';
import { VoiceEngineConfig } from '../config';
import { ssmParams } from '../ssm-parameters';
import { SecretsConstruct, RecordingsBucketConstruct } from '../constructs';

export interface StorageStackProps extends cdk.StackProps {
  readonly config: VoiceEngineConfig;
}

/**
 * Storage stack — owns Secrets Manager entries and the recordings
 * bucket. Publishes every output as an SSM parameter; downstream stacks
 * (ComputeStack in Wave 3) read via `ssm.StringParameter.valueFromLookup`.
 *
 * The two sub-constructs (`SecretsConstruct`, `RecordingsBucketConstruct`)
 * each encapsulate their own per-env decisions. This stack just composes
 * them and exposes the publication surface.
 */
export class StorageStack extends cdk.Stack {
  public readonly secrets: SecretsConstruct;
  public readonly recordings: RecordingsBucketConstruct;

  constructor(scope: Construct, id: string, props: StorageStackProps) {
    super(scope, id, props);
    const { config } = props;

    this.secrets = new SecretsConstruct(this, 'Secrets', { config });
    this.recordings = new RecordingsBucketConstruct(this, 'Recordings', { config });

    const params = ssmParams(config);

    new ssm.StringParameter(this, 'ApiKeySecretArnParam', {
      parameterName: params.API_KEY_SECRET_ARN,
      stringValue: this.secrets.apiKeySecretArn,
      description: 'ARN of the HTTP bearer auth secret for /start.',
    });
    new ssm.StringParameter(this, 'DailyApiKeySecretArnParam', {
      parameterName: params.DAILY_API_KEY_SECRET_ARN,
      stringValue: this.secrets.dailyApiKeySecretArn,
      description: 'ARN of the Daily.co REST API key secret.',
    });
    new ssm.StringParameter(this, 'AssemblyAiApiKeySecretArnParam', {
      parameterName: params.ASSEMBLYAI_API_KEY_SECRET_ARN,
      stringValue: this.secrets.assemblyAiApiKeySecretArn,
      description: 'ARN of the AssemblyAI API key secret.',
    });
    new ssm.StringParameter(this, 'ElevenLabsApiKeySecretArnParam', {
      parameterName: params.ELEVENLABS_API_KEY_SECRET_ARN,
      stringValue: this.secrets.elevenLabsApiKeySecretArn,
      description: 'ARN of the ElevenLabs API key secret.',
    });

    new ssm.StringParameter(this, 'RecordingsBucketNameParam', {
      parameterName: params.RECORDINGS_BUCKET_NAME,
      stringValue: this.recordings.bucketName,
      description: 'S3 bucket name for call recordings.',
    });
    new ssm.StringParameter(this, 'RecordingsBucketArnParam', {
      parameterName: params.RECORDINGS_BUCKET_ARN,
      stringValue: this.recordings.bucketArn,
      description: 'S3 bucket ARN for call recordings.',
    });

    if (this.recordings.kmsKeyArn) {
      new ssm.StringParameter(this, 'RecordingsKmsKeyArnParam', {
        parameterName: params.RECORDINGS_KMS_KEY_ARN,
        stringValue: this.recordings.kmsKeyArn,
        description:
          'KMS key ARN protecting the recordings bucket. For prod, this ' +
          'requires .env.prod to set RECORDINGS_KMS_KEY_ARN; if unset, the ' +
          'parameter is not created and ComputeStack will need a different ' +
          'fallback.',
      });
    }

    new cdk.CfnOutput(this, 'RecordingsBucketName', {
      value: this.recordings.bucketName,
    });
    new cdk.CfnOutput(this, 'RecordingsBucketOwnedByCdk', {
      value: String(this.recordings.ownedByCdk),
    });
    new cdk.CfnOutput(this, 'RecordingsKmsKeyArn', {
      value: this.recordings.kmsKeyArn || '(not set — populate .env.prod for prod)',
    });
  }
}
