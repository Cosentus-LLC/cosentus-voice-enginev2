import * as cdk from 'aws-cdk-lib';
import { CfnSecret } from 'aws-cdk-lib/aws-secretsmanager';
import { Construct } from 'constructs';
import { VoiceEngineConfig } from '../config';

export interface SecretsConstructProps {
  readonly config: VoiceEngineConfig;
}

/**
 * Four Secrets Manager entries the engine task pulls at startup.
 *
 * Why L1 `CfnSecret` instead of L2 `Secret`
 * -----------------------------------------
 * The L2 `secretsmanager.Secret` construct always seeds an initial
 * version (either generated, or set via `secretStringValue`). Per the
 * Wave 2 brief we want truly empty entries — the operator fills them in
 * via the AWS Console before deploy. AWS allows secret metadata to
 * exist without an initial version; CFN supports it by simply omitting
 * `SecretString`/`GenerateSecretString` on `AWS::SecretsManager::Secret`.
 * The L1 `CfnSecret` exposes that path; the L2 does not.
 *
 * Engine behavior with empty secrets
 * ----------------------------------
 * `GetSecretValue` against a secret with no versions returns
 * `ResourceNotFoundException`. Layer 2 (`app/config/settings.py`) loads
 * each key at startup and surfaces the failure as a startup crash, which
 * is the intended fail-fast — we never run a task with a half-configured
 * key set.
 *
 * Naming
 * ------
 * `cosentus-voice-engine/{environment}/{secret-name}`. The brief used
 * the unscoped `voice-engine/{name}` form, but a multi-env deployment
 * requires per-env scoping; the prefix change is documented in
 * README.md.
 *
 * Lifecycle
 * ---------
 * RETAIN on both deletion and update-replace. Recordings of API key
 * history is intentional; if we destroy a stack we don't want the
 * secret entries (and their populated values) to vanish.
 */
export class SecretsConstruct extends Construct {
  public readonly apiKeySecretArn: string;
  public readonly dailyApiKeySecretArn: string;
  public readonly assemblyAiApiKeySecretArn: string;
  public readonly elevenLabsApiKeySecretArn: string;

  constructor(scope: Construct, id: string, props: SecretsConstructProps) {
    super(scope, id);
    const namespace = `cosentus-voice-engine/${props.config.environment}`;

    const apiKey = this.makeSecret('ApiKeySecret', {
      name: `${namespace}/api-key`,
      description:
        'HTTP bearer auth for /start. Populate via AWS Console before deploy. ' +
        'Created with no initial version by CDK.',
    });

    const dailyApiKey = this.makeSecret('DailyApiKeySecret', {
      name: `${namespace}/daily-api-key`,
      description:
        'Daily.co REST API key for room/token/dial-out management. Populate ' +
        'via AWS Console before deploy.',
    });

    const assemblyAiApiKey = this.makeSecret('AssemblyAiApiKeySecret', {
      name: `${namespace}/assemblyai-api-key`,
      description:
        'AssemblyAI Universal-Streaming v3 API key. Populate via AWS Console ' +
        'before deploy.',
    });

    const elevenLabsApiKey = this.makeSecret('ElevenLabsApiKeySecret', {
      name: `${namespace}/elevenlabs-api-key`,
      description:
        'ElevenLabs TTS API key. Populate via AWS Console before deploy.',
    });

    this.apiKeySecretArn = apiKey.ref;
    this.dailyApiKeySecretArn = dailyApiKey.ref;
    this.assemblyAiApiKeySecretArn = assemblyAiApiKey.ref;
    this.elevenLabsApiKeySecretArn = elevenLabsApiKey.ref;
  }

  private makeSecret(
    id: string,
    props: { name: string; description: string },
  ): CfnSecret {
    const secret = new CfnSecret(this, id, {
      name: props.name,
      description: props.description,
    });
    secret.cfnOptions.deletionPolicy = cdk.CfnDeletionPolicy.RETAIN;
    secret.cfnOptions.updateReplacePolicy = cdk.CfnDeletionPolicy.RETAIN;
    return secret;
  }
}
