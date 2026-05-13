import * as path from 'path';
import * as cdk from 'aws-cdk-lib';
import * as iam from 'aws-cdk-lib/aws-iam';
import * as lambda from 'aws-cdk-lib/aws-lambda';
import * as logs from 'aws-cdk-lib/aws-logs';
import { Construct } from 'constructs';
import { VoiceEngineConfig } from '../config';

export interface VoiceApiStubStackProps extends cdk.StackProps {
  readonly config: VoiceEngineConfig;
}

/**
 * Staging-only stub of the production `cosentus-voice-api` Lambda.
 *
 * Why this exists
 * ---------------
 * The engine's Layer 1 + Layer 6 invoke the cosentus-voice-api Lambda
 * for runtime-config reads and call-record writes. Pointing staging at
 * the production Lambda was rejected during Wave 7 deploy review:
 * the read-only runtime-config path is fine, but call-record writes
 * would land in production Aurora. The stub eliminates that risk by
 * returning canned responses for every operation the engine invokes.
 *
 * Lifetime
 * --------
 * Temporary. When Phase 6 stands up a real staging cosentus-voice-api
 * Lambda alias backed by a staging Aurora schema, destroy this stack
 * and repoint `VOICE_API_LAMBDA_NAME` at the real one. The engine's
 * code doesn't change.
 *
 * Lambda name shape
 * -----------------
 * `cosentus-voice-api-{env}-stub`. The compute stack's IAM policy
 * grants `lambda:InvokeFunction` on `cosentus-voice-api-*`, which
 * matches this name (and would match a future prod stub if we ever
 * needed one).
 *
 * Cost
 * ----
 * Free until invoked. Each call adds ~3 invocations (runtime-config
 * + 1 or 2 write-call-record + auto-actions). Even at Wave 6's
 * highest concurrency tier (N=6 calls/min over 30 min), that's
 * <2000 invocations — well under the Lambda free tier's monthly
 * 1M. Reserved concurrency intentionally unset; small staging load
 * doesn't justify the operational complexity.
 */
export class VoiceApiStubStack extends cdk.Stack {
  public readonly functionName: string;
  public readonly functionArn: string;

  constructor(scope: Construct, id: string, props: VoiceApiStubStackProps) {
    super(scope, id, props);
    const { config } = props;

    const functionName = `cosentus-voice-api-${config.environment}-stub`;

    const logGroup = new logs.LogGroup(this, 'LogGroup', {
      logGroupName: `/aws/lambda/${functionName}`,
      retention: logs.RetentionDays.ONE_MONTH,
      removalPolicy: cdk.RemovalPolicy.DESTROY,
    });

    const role = new iam.Role(this, 'ExecutionRole', {
      roleName: `${functionName}-role`,
      assumedBy: new iam.ServicePrincipal('lambda.amazonaws.com'),
      description: 'Voice-api stub Lambda execution role (logs only).',
      managedPolicies: [
        iam.ManagedPolicy.fromAwsManagedPolicyName('service-role/AWSLambdaBasicExecutionRole'),
      ],
    });

    const fn = new lambda.Function(this, 'Function', {
      functionName,
      runtime: lambda.Runtime.PYTHON_3_12,
      architecture: lambda.Architecture.X86_64,
      code: lambda.Code.fromAsset(path.join(__dirname, '..', '..', 'lambdas', 'voice-api-stub')),
      handler: 'handler.handler',
      memorySize: 256,
      timeout: cdk.Duration.seconds(10),
      role,
      logGroup,
      description: (
        `Staging-only stub for cosentus-voice-api. Returns canned runtime-config ` +
        `and accepts call-record / auto-actions writes as no-ops. Decommission ` +
        `when Phase 6 ships the real staging Lambda alias.`
      ),
      environment: {
        STUB_ENVIRONMENT: config.environment,
      },
    });
    this.functionName = fn.functionName;
    this.functionArn = fn.functionArn;

    new cdk.CfnOutput(this, 'FunctionName', {
      value: fn.functionName,
      description: 'Set as VOICE_API_LAMBDA_NAME on the staging compute task.',
    });
    new cdk.CfnOutput(this, 'FunctionArn', { value: fn.functionArn });
  }
}
