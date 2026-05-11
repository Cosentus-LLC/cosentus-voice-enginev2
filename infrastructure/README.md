# Voice engine v2 — AWS CDK infrastructure

TypeScript CDK stacks for deploying the v2 voice engine to AWS Fargate
behind an internet-facing ALB. Staging + prod environments are deployable
independently from one codebase.

## Status

Wave 1 of 5 — scaffold + `EcrStack` + `NetworkStack`.

| Wave | Scope | Status |
|------|-------|--------|
| 1 | Skeleton + ECR + VPC + network | shipped |
| 2 | `StorageStack` (Secrets Manager + S3 recordings) | pending |
| 3 | `CertStack` (wildcard ACM) + `ComputeStack` (ALB + ECS + autoscaling) | pending |
| 4 | Engine-side `cloudwatch:PutMetricData` for `ActiveSessions` | pending |
| 5 | GitHub Actions (`build-and-push`, `deploy-staging`, `deploy-prod`) + doc cleanup | pending |

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
    constructs/
      vpc.ts                     # VPC + 8 endpoints + 2 security groups
```

## Locked-in choices

| Knob | Value | Source |
|------|-------|--------|
| AWS account | `825269749545` | v1 production account (same one) |
| Region | `us-east-1` | Bedrock inference profiles + Daily PSTN trunk |
| Task sizing | 1 vCPU / 2 GB / `stopTimeout=120s` | Layer 9.5 scale test |
| Max concurrent calls / task | 6 | Layer 9.5 + Bug D capacity gate |
| ALB scheme | internet-facing, HTTPS only | Daily webhook lands publicly |
| ACM strategy | one shared wildcard `*.cosentusaibackend.com`, DNS-validated | minimizes GoDaddy DNS work |
| ECR | dedicated repo `cosentus-voice-engine`, immutable tags | CI/CD push path, not CDK asset |
| NAT Gateways | staging=1, prod=3 | cost vs HA per env |
| VPC CIDRs | staging `10.20.0.0/16`, prod `10.30.0.0/16` | non-overlapping with v1's `vpc-05a1f6c68c04943ec` |

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
| `RECORDINGS_BUCKET_NAME` | `cosentus-voice-engine-staging-recordings` (CDK creates) | `medcloud-voice-us-prod-825` (imported, owned by v1) |

`config.ts` carries the same defaults so deploying without a `.env`
file still produces correct synthesized CloudFormation.

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
