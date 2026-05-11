import * as cdk from 'aws-cdk-lib';
import * as kms from 'aws-cdk-lib/aws-kms';
import * as s3 from 'aws-cdk-lib/aws-s3';
import { Construct } from 'constructs';
import { VoiceEngineConfig, resourcePrefix } from '../config';

export interface RecordingsBucketConstructProps {
  readonly config: VoiceEngineConfig;
}

/**
 * S3 recordings bucket — create-or-import per environment.
 *
 * Strategy
 * --------
 *   - Staging (`recordingsBucketOwnedByCdk=true`): CDK creates a fresh
 *     bucket + a customer-managed KMS key. Full ownership of lifecycle
 *     and encryption. Cheap to tear down for resets.
 *   - Prod   (`recordingsBucketOwnedByCdk=false`): CDK imports
 *     `medcloud-voice-us-prod-825` (v1's existing bucket — already wired
 *     into Daily.co, already populated with production recordings). CDK
 *     does not declare the bucket as a resource, so destroying this
 *     stack will not touch it. ComputeStack (Wave 3) reads the bucket
 *     ARN + KMS key ARN via SSM and grants the task role IAM permissions
 *     against them; CDK never owns the bucket policy.
 *
 * Why a single construct for both paths
 * -------------------------------------
 * StorageStack treats "recordings bucket" as one logical thing. The
 * branch on `ownedByCdk` keeps the per-env divergence local and the
 * surface area uniform: callers get `bucketName`, `bucketArn`,
 * `kmsKeyArn`, and `ownedByCdk` regardless of which path ran.
 *
 * KMS key (staging path)
 * ----------------------
 *   - Alias `alias/cosentus-voice-engine-{env}-recordings`.
 *   - Key rotation enabled (HIPAA-adjacent data path; annual rotation
 *     is a sensible default).
 *   - Removal policy RETAIN — under no circumstance should
 *     `cdk destroy` shred the key that encrypts call recordings.
 *
 * Bucket hardening (staging path)
 * -------------------------------
 *   - Block all public access (account-level + bucket-level).
 *   - Bucket-owner-enforced ACLs (no legacy ACL grants).
 *   - Bucket-key enabled (reduces KMS API call cost on heavy traffic).
 *   - Versioning enabled (recordings audit trail).
 *   - SSL-only via `enforceSSL` (auto-attached `aws:SecureTransport=true`
 *     deny policy).
 *   - Removal policy RETAIN.
 *
 * Daily.co write path (TODO for Wave 3 follow-up)
 * -----------------------------------------------
 * Daily writes recordings server-side via a configured bucket and KMS
 * key in the Daily dashboard. The bucket policy needs to allow Daily's
 * principal to PutObject + GenerateDataKey. We defer that bucket-policy
 * statement to ComputeStack alongside the task-role grants, because
 * both write paths share the same "who can touch this bucket" decision
 * and should be reviewed together.
 */
export class RecordingsBucketConstruct extends Construct {
  /** S3 bucket name. */
  public readonly bucketName: string;
  /** S3 bucket ARN. */
  public readonly bucketArn: string;
  /**
   * KMS key ARN. Always present for staging (CDK-created). For prod
   * comes from `config.recordingsKmsKeyArn`; may be empty until the
   * operator populates `.env.prod`.
   */
  public readonly kmsKeyArn: string;
  /** True when CDK owns the bucket lifecycle. */
  public readonly ownedByCdk: boolean;

  constructor(scope: Construct, id: string, props: RecordingsBucketConstructProps) {
    super(scope, id);
    const { config } = props;
    this.ownedByCdk = config.recordingsBucketOwnedByCdk;

    if (this.ownedByCdk) {
      const prefix = resourcePrefix(config);

      const key = new kms.Key(this, 'RecordingsKey', {
        alias: `alias/${prefix}-recordings`,
        description: `SSE-KMS key for v2 ${config.environment} call recordings. CDK-owned.`,
        enableKeyRotation: true,
        removalPolicy: cdk.RemovalPolicy.RETAIN,
      });

      const bucket = new s3.Bucket(this, 'RecordingsBucket', {
        bucketName: config.recordingsBucketName,
        encryption: s3.BucketEncryption.KMS,
        encryptionKey: key,
        bucketKeyEnabled: true,
        blockPublicAccess: s3.BlockPublicAccess.BLOCK_ALL,
        objectOwnership: s3.ObjectOwnership.BUCKET_OWNER_ENFORCED,
        versioned: true,
        enforceSSL: true,
        removalPolicy: cdk.RemovalPolicy.RETAIN,
      });

      this.bucketName = bucket.bucketName;
      this.bucketArn = bucket.bucketArn;
      this.kmsKeyArn = key.keyArn;
    } else {
      this.bucketName = config.recordingsBucketName;
      this.bucketArn = cdk.Stack.of(this).formatArn({
        service: 's3',
        region: '',
        account: '',
        resource: config.recordingsBucketName,
      });
      this.kmsKeyArn = config.recordingsKmsKeyArn;
    }
  }
}
