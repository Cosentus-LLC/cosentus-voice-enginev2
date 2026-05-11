import * as cdk from 'aws-cdk-lib';
import * as ecr from 'aws-cdk-lib/aws-ecr';
import * as ssm from 'aws-cdk-lib/aws-ssm';
import { Construct } from 'constructs';
import { VoiceEngineConfig } from '../config';
import { ssmParams } from '../ssm-parameters';

export interface EcrStackProps extends cdk.StackProps {
  readonly config: VoiceEngineConfig;
}

/**
 * Dedicated ECR repository for the voice engine container image.
 *
 * Why dedicated (not CDK asset bucket)
 * ------------------------------------
 * v1's EcsStack built its image via `DockerImageAsset` and pushed to the
 * CDK-bootstrapped asset bucket. That pattern couples the image lifecycle
 * to `cdk deploy` and makes CI/CD (GitHub Actions building + pushing on
 * `main` merge) awkward.
 *
 * v2 separates the planes: this stack owns the repo, CI/CD pushes images
 * to it, and ComputeStack references images by `repoUri:tag`. The image
 * tag is the per-env deployment knob.
 *
 * Lifecycle policy
 * ----------------
 * Keep the 30 most-recent untagged images (cleanup of dangling CI-built
 * layers) and the 50 most-recent tagged images (covers rollback range
 * comfortably). Anything older auto-deletes. Keeps storage cost bounded
 * while preserving rollback ability.
 *
 * Note on env-independence
 * ------------------------
 * The repo is created in the staging or prod stack — whichever runs first
 * — but the repo itself is shared by both. `EcrStack` is therefore env-
 * scoped *by stack name* but env-shared *by underlying resource*. To
 * avoid a double-create on second-env synth, we set `imported` semantics
 * via CloudFormation's `Retain` removal policy: both stacks declare the
 * same repo name; only the first `cdk deploy` creates it; subsequent
 * deploys are no-ops on this resource.
 *
 * Operational note: when destroying the *first* env's stack, AWS will
 * refuse to delete the repo if the second env's stack still references it
 * via SSM. That's intentional — protects against accidental teardown of
 * the prod image registry by a staging destroy.
 */
export class EcrStack extends cdk.Stack {
  public readonly repository: ecr.IRepository;

  constructor(scope: Construct, id: string, props: EcrStackProps) {
    super(scope, id, props);
    const { config } = props;

    const repoName = config.projectName;

    const repository = new ecr.Repository(this, 'Repository', {
      repositoryName: repoName,
      imageTagMutability: ecr.TagMutability.IMMUTABLE,
      imageScanOnPush: true,
      encryption: ecr.RepositoryEncryption.AES_256,
      lifecycleRules: [
        {
          rulePriority: 1,
          description: 'Expire untagged images after 7 days',
          tagStatus: ecr.TagStatus.UNTAGGED,
          maxImageAge: cdk.Duration.days(7),
        },
        {
          rulePriority: 2,
          description: 'Retain only the 50 most recent tagged images',
          tagStatus: ecr.TagStatus.ANY,
          maxImageCount: 50,
        },
      ],
      removalPolicy: cdk.RemovalPolicy.RETAIN,
    });
    this.repository = repository;

    const params = ssmParams(config);
    new ssm.StringParameter(this, 'EcrRepositoryUriParam', {
      parameterName: params.ECR_REPOSITORY_URI,
      stringValue: repository.repositoryUri,
      description: 'ECR repository URI for the voice engine container image.',
      tier: ssm.ParameterTier.STANDARD,
    });
    new ssm.StringParameter(this, 'EcrRepositoryArnParam', {
      parameterName: params.ECR_REPOSITORY_ARN,
      stringValue: repository.repositoryArn,
      description: 'ECR repository ARN for the voice engine container image.',
      tier: ssm.ParameterTier.STANDARD,
    });

    new cdk.CfnOutput(this, 'RepositoryUri', {
      value: repository.repositoryUri,
      exportName: `${id}-RepositoryUri`,
      description: 'Push images to this URI from CI/CD.',
    });
    new cdk.CfnOutput(this, 'RepositoryName', {
      value: repository.repositoryName,
    });
  }
}
