# Voice engine v2 — AWS CDK infrastructure

TypeScript CDK stacks for deploying the v2 voice engine to AWS Fargate
behind an internet-facing ALB. Staging + prod environments are deployable
independently from one codebase.

## Status

All foundational CDK shipped. Staging deployed end-to-end. Wave 6 mock
load test is the next workstream (separate harness, not CDK).

| Wave | Scope | Status |
|------|-------|--------|
| 1 | Skeleton + ECR + VPC + network | shipped |
| 2 | `StorageStack` (Secrets Manager + S3 recordings + KMS) | shipped |
| 3 | `CertStack` (wildcard ACM) + `ComputeStack` (ALB + ECS + autoscaling + monitoring) | shipped |
| 4 | Engine-side `cloudwatch:PutMetricData` for `ActiveSessions` + `DrainTimeouts` | shipped |
| 5 | GitHub Actions (`build-and-push`, `deploy-staging`, `deploy-prod`) + doc cleanup | deferred to Phase 5 (post-prod) |
| 7 | Staging deploy + `VoiceApiStubStack` for prod-data isolation during testing | shipped |

## Quick start

```bash
cd infrastructure
npm install
cp .env.example .env.staging  # then edit secrets / per-env values
npx cdk synth -c environment=staging
```

`.env.staging` and `.env.prod` are git-ignored. Never commit secrets.

## Layout

```
infrastructure/
  package.json
  tsconfig.json
  cdk.json
  .env.example                   # copy to .env.staging / .env.prod
  src/
    main.ts                      # per-env stack composition
    config.ts                    # env validation + defaults
    ssm-parameters.ts            # cross-stack SSM keys
    stacks/
      ecr-stack.ts               # dedicated ECR repo (env-independent)
      network-stack.ts           # VPC, subnets, security groups, endpoints
      storage-stack.ts           # Secrets Manager + recordings bucket
      cert-stack.ts              # shared wildcard ACM (env-independent)
      compute-stack.ts           # ALB + ECS + autoscaling + monitoring
      voice-api-stub-stack.ts    # staging-only stub of cosentus-voice-api Lambda
    constructs/
      vpc.ts                     # VPC + 8 endpoints + 2 security groups
      secrets.ts                 # 4 empty Secrets Manager entries (L1 CfnSecret)
      recordings-bucket.ts       # S3 + KMS (create) or import per env
      alb.ts                     # internet-facing ALB + HTTPS listener + target group
      ecs-service.ts             # cluster + task def + Fargate service + autoscaling
      monitoring.ts              # SNS alarm topic + 5 alarms + dashboard
  lambdas/
    voice-api-stub/
      handler.py                 # canned runtime-config + no-op writes
```

## Locked-in choices

| Knob | Value | Source |
|------|-------|--------|
| AWS account | `825269749545` | v1 production account (same one) |
| Region | `us-east-1` | Bedrock inference profiles + Daily PSTN trunk |
| Task sizing | 1 vCPU / 2 GB / `stopTimeout=120s` | Layer 9.5 scale test |
| Max concurrent calls / task | 6 | Layer 9.5 + Bug D capacity gate |
| ALB scheme | internet-facing, HTTPS only, TLS 1.2+ | Daily webhook lands publicly |
| ACM strategy | one shared wildcard `*.cosentusaibackend.com`, DNS-validated | minimizes GoDaddy DNS work |
| ECR | dedicated repo `cosentus-voice-engine`, immutable tags | CI/CD push path, not CDK asset |
| NAT Gateways | staging=1, prod=3 | cost vs HA per env |
| VPC CIDRs | staging `10.20.0.0/16`, prod `10.30.0.0/16` | non-overlapping with v1's `vpc-05a1f6c68c04943ec` |
| Autoscaling | target tracking on `ActiveSessions`, target=4.2/task (70% of 6), out=60s, in=300s | brief + validate-then-commit prod sizing |
| Capacity | staging min=1/max=5, prod min=1/max=25 | start small, raise after Wave 6 mock load |

## DNS / TLS path

`cosentusaibackend.com` is registered at **GoDaddy** (nameservers
`ns77/ns78.domaincontrol.com`). v2 does **not** delegate to Route 53 —
DNS stays at GoDaddy and we add two records manually:

1. **ACM validation CNAME** — one-shot, set per the values ACM emits
   when CertStack first creates the cert.
2. **ALB target CNAME** — for `api.cosentusaibackend.com` (prod) and
   `staging.cosentusaibackend.com` (staging), each CNAMEd to the ALB's
   DNS name from `cdk deploy` outputs.

This avoids migrating the apex (and its existing records like
`portal.cosentus.com` peers) into Route 53.

## Per-env values (`.env.staging` vs `.env.prod`)

| Variable | Staging | Production |
|----------|---------|------------|
| `ENVIRONMENT` | `staging` | `prod` |
| `SERVICE_HOSTNAME` | `staging.cosentusaibackend.com` | `api.cosentusaibackend.com` |
| `MIN_CAPACITY` | 1 | 5 |
| `MAX_CAPACITY` | 3 | 25 |
| `NAT_GATEWAYS` | 1 | 3 |
| `RECORDINGS_BUCKET_NAME` | `cosentus-voice-recordings-staging` (CDK creates) | `medcloud-voice-us-prod-825` (imported, owned by v1) |
| `RECORDINGS_KMS_KEY_ARN` | auto-set by CDK at synth | required in `.env.prod` (v1's existing key ARN) |

`config.ts` carries the same defaults so deploying without a `.env`
file still produces correct synthesized CloudFormation.

## Deploy order (Wave 3+)

ComputeStack reads its inputs via SSM dynamic refs (`{{resolve:ssm:/...}}`),
so synth always succeeds from scratch — but deploys fail loudly until
the upstream stacks have published their SSM values. The required
sequence per environment:

```
1. cdk deploy cosentus-voice-engine-cert                       # one-time, env-independent
                                                               # then: add CNAME at GoDaddy,
                                                               # wait for ACM validation
2. cdk deploy cosentus-voice-engine-<env>-ecr                  # creates the ECR repo
3. (push the engine image, tag it, get a sha)                  # CI/CD or manual
4. cdk deploy cosentus-voice-engine-<env>-network              # VPC, NAT, endpoints
5. cdk deploy cosentus-voice-engine-<env>-storage              # secrets (empty) + bucket
6. (populate secrets via AWS Console / CLI, see below)         # API keys go in plaintext
7. cdk deploy cosentus-voice-engine-staging-voice-api-stub     # staging only
8. cdk deploy cosentus-voice-engine-<env>-compute              # ALB + ECS + autoscaling
                                                               # then: add CNAME at GoDaddy
                                                               # pointing serviceHostname →
                                                               # ALB DNS output
```

After step 8, subscribe operator endpoints (email / PagerDuty / Slack
incoming-webhook bridge) to the SNS topic published as
`AlarmTopicArn` in ComputeStack's outputs.

### Why staging has a voice-api stub Lambda

The engine's Layer 1 + Layer 6 invoke the cosentus-voice-api Lambda
for runtime-config reads and call-record writes. Production's Lambda
(`medcloud-voice-api:live`) writes to production Aurora. Staging tests
should not.

`VoiceApiStubStack` deploys `cosentus-voice-api-staging-stub`, a tiny
Python Lambda that:

- Returns a canned `AgentConfig` for any `agent_id` (Haiku model,
  `press_digit` + `end_call` tools, generic system prompt).
- Accepts `POST /api/calls` and `POST /api/auto-actions` as 200 no-ops.
- Logs each invocation as JSON for CloudWatch Insights.

The engine's `VOICE_API_LAMBDA_NAME` for staging defaults to this
stub (see `config.ts`). Phase 6 will deploy a real staging Lambda
alias against a staging Aurora schema; at that point destroy
`VoiceApiStubStack` and repoint the env var.

**Shell gotcha noted during Wave 7 deploy.** Don't `source .env.skeleton`
in the same shell you run `cdk deploy` from — it exports
`VOICE_API_LAMBDA_NAME` and other vars into the shell, and CDK's
config.ts will read those env-var overrides instead of the
per-env defaults. If you must source it (e.g., to populate secrets
via CLI), spawn a subshell with `bash -c '...'` for the deploy
or `unset VOICE_API_LAMBDA_NAME API_KEY_SECRET_ARN DAILY_API_KEY
ASSEMBLYAI_API_KEY ELEVENLABS_API_KEY` first.

## Secrets Manager — populating before deploy

Wave 2 creates four Secrets Manager entries per environment, **all
empty** (no initial version). The engine task fails fast at startup if
any of them returns `ResourceNotFoundException`, so the operator must
populate values via the AWS Console before the first task launch.

Canonical secret names:

```
cosentus-voice-engine/{env}/api-key             # HTTP bearer auth for /start
cosentus-voice-engine/{env}/daily-api-key       # Daily.co REST API
cosentus-voice-engine/{env}/assemblyai-api-key  # AssemblyAI v3 streaming
cosentus-voice-engine/{env}/elevenlabs-api-key  # ElevenLabs TTS
```

To populate a secret:

1. AWS Console → Secrets Manager → pick the entry by name.
2. **Set secret value** → **Plaintext** → paste the API key as the entire
   value (no JSON wrapping; the engine reads `SecretString` as a raw string).
3. Save. Confirm a new `AWSCURRENT` version is listed.

Lifecycle: each secret has `RETAIN` set on both deletion and update-replace,
so destroying the stack leaves the entries (and their populated values)
intact — intentional to avoid accidentally re-keying production credentials.

## Recordings bucket

| Knob | Staging | Production |
|------|---------|------------|
| Bucket name | `cosentus-voice-recordings-staging` | `medcloud-voice-us-prod-825` |
| Lifecycle | CDK-owned (create) | Imported (CDK touches IAM only) |
| KMS key | CDK-created, `alias/cosentus-voice-engine-staging-recordings`, rotation enabled | Existing v1 key (ARN must be set in `.env.prod`) |
| Public access | blocked at bucket level | (out of CDK's scope — unchanged) |
| Versioning | enabled | (unchanged) |
| SSL-only | enforced | (unchanged) |

Daily.co's write principal still needs to be granted access via the
bucket policy. That statement is deferred to Wave 3 (ComputeStack) so
it lives alongside the task-role grants.

## Deployment workflows (planned, Wave 5)

- `.github/workflows/build-and-push.yml` — on push to `main`: build the
  image, tag with `<git-sha>` and `staging-latest`, push to ECR.
- `.github/workflows/deploy-staging.yml` — auto-deploy to staging after
  successful build-and-push.
- `.github/workflows/deploy-prod.yml` — manual workflow dispatch with
  GitHub environment protection rules (approver required).

## Tests (planned)

`test/` will hold CDK assertion tests per stack
(`@aws-cdk/assertions`), focused on:
- Stack synthesizes for both `staging` and `prod` contexts without errors.
- ECR repo enforces immutable tags + scan-on-push.
- Network stack creates the expected NAT count per env.
- ALB is HTTPS-only, no port-80 listener.
- IAM task role grants exactly the actions in `docs/architecture/overview.md`,
  not the union of every AWS managed policy.

## References

- `docs/architecture/overview.md` — system spec.
- `docs/architecture/migration-from-v1.md` — layer-by-layer plan.
- `docs/v2-tech-debt-log.md` — entry 13 (AssemblyAI 1008), entry 12 (signal
  handlers), and others.
- v1 CDK at `~/Desktop/cosentus-voice-engine/infrastructure/` — read-only
  reference, not source of truth.
