#!/usr/bin/env node
/**
 * CDK entry point for the v2 voice engine.
 *
 * Composes per-environment stacks. The environment is resolved at synth
 * time via `cdk synth -c environment=staging|prod` (or the ENVIRONMENT
 * env var). Synthing one env does not touch the other — staging and prod
 * are deployable independently.
 *
 * Stack composition (deployment order enforced by addDependency)
 * --------------------------------------------------------------
 *
 *   1. EcrStack         (env-independent; created once, both envs pull
 *                        the same repo, differentiated by image tag)
 *   2. NetworkStack     (VPC, subnets, security groups, VPC endpoints)
 *   3. StorageStack     (Secrets Manager + recordings bucket reference)   — Wave 2
 *   4. CertStack        (shared wildcard ACM cert + DNS validation)        — Wave 3
 *   5. ComputeStack     (ALB, ECS cluster, Fargate service, monitoring)    — Wave 3
 *
 * Cross-stack references flow through SSM parameters (see ssm-parameters.ts)
 * to avoid CloudFormation cyclic dependencies.
 */

import 'source-map-support/register';
import * as cdk from 'aws-cdk-lib';
import { loadConfig, resourcePrefix } from './config';
import {
  CertStack,
  ComputeStack,
  EcrStack,
  NetworkStack,
  StorageStack,
  VoiceApiStubStack,
} from './stacks';

function resolveImageTag(app: cdk.App): string {
  const fromContext = app.node.tryGetContext('imageTag');
  if (typeof fromContext === 'string' && fromContext.length > 0) return fromContext;
  const fromEnv = process.env.IMAGE_TAG;
  if (typeof fromEnv === 'string' && fromEnv.length > 0) return fromEnv;
  return 'latest';
}

const app = new cdk.App();
const config = loadConfig(app);

const env: cdk.Environment = {
  account: config.account,
  region: config.region,
};

const commonTags: Record<string, string> = {
  Project: config.projectName,
  Environment: config.environment,
  ManagedBy: 'cdk',
};

const prefix = resourcePrefix(config);

const ecrStack = new EcrStack(app, `${prefix}-ecr`, {
  env,
  config,
  description: 'Voice engine v2 — dedicated ECR repository (env-independent).',
  tags: { ...commonTags, Stack: 'ecr' },
});

const networkStack = new NetworkStack(app, `${prefix}-network`, {
  env,
  config,
  description: `Voice engine v2 — VPC, subnets, endpoints (${config.environment}).`,
  tags: { ...commonTags, Stack: 'network' },
});

const storageStack = new StorageStack(app, `${prefix}-storage`, {
  env,
  config,
  description: `Voice engine v2 — Secrets Manager + recordings bucket (${config.environment}).`,
  tags: { ...commonTags, Stack: 'storage' },
});

const certStack = new CertStack(app, 'cosentus-voice-engine-cert', {
  env,
  config,
  description:
    'Voice engine v2 — wildcard ACM cert covering both env hostnames ' +
    '(shared, env-independent stack name).',
  tags: { ...commonTags, Stack: 'cert', Shared: 'true' },
});

const imageTag = resolveImageTag(app);

// Voice-api stub Lambda — staging only. Stands in for the real
// cosentus-voice-api Lambda so the staging engine can run end-to-end
// calls without writing to production Aurora. Destroy when the real
// staging Lambda alias lands in Phase 6.
let voiceApiStubStack: VoiceApiStubStack | undefined;
if (config.environment === 'staging') {
  voiceApiStubStack = new VoiceApiStubStack(app, `${prefix}-voice-api-stub`, {
    env,
    config,
    description: `Voice engine v2 — voice-api stub Lambda (${config.environment}).`,
    tags: { ...commonTags, Stack: 'voice-api-stub' },
  });
}

const computeStack = new ComputeStack(app, `${prefix}-compute`, {
  env,
  config,
  imageTag,
  description: `Voice engine v2 — ALB + ECS Fargate + autoscaling + monitoring (${config.environment}).`,
  tags: { ...commonTags, Stack: 'compute' },
});

// All cross-stack data flows through SSM (deploy-time CFN dynamic refs).
// No direct CFN dependency arrows; ordering is enforced by the deploy
// sequence documented in infrastructure/README.md.
void ecrStack;
void networkStack;
void storageStack;
void certStack;
void computeStack;
void voiceApiStubStack;

app.synth();
