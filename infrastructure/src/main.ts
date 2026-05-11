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
import { EcrStack, NetworkStack, StorageStack } from './stacks';

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

// Stacks publish to SSM and are read via valueFromLookup downstream;
// no direct CFN-level dependency arrows are required here.
void ecrStack;
void networkStack;
void storageStack;

app.synth();
