# Deploying the InfoLang AgentCore agent

Two supported paths. Both give the same result: an AgentCore Runtime whose
memory is durable because it lives in InfoLang, keyed by actor identity — so it
survives session teardown/recreate and account reprovisioning.

## Option A — Starter toolkit (fastest)

```bash
pip install bedrock-agentcore-starter-toolkit
cd examples

agentcore configure --entrypoint agent.py     # generates Dockerfile + config
agentcore launch                               # builds arm64 image, pushes ECR,
                                               # creates the Runtime + IAM role
```

Provide the InfoLang API key as an environment variable / secret when prompted.

## Option B — Terraform (per-customer account / WP17)

For the dedicated-AWS-account Enterprise model, manage the Runtime as IaC:

```bash
# 1. Build + push the arm64 image (from examples/).
docker buildx build --platform linux/arm64 -t "$IMAGE_URI" --push .

# 2. Store the InfoLang key.
aws secretsmanager create-secret --name infolang-api-key --secret-string "$INFOLANG_API_KEY"

# 3. Apply.
cd deploy
terraform init
terraform apply \
  -var "image_uri=$IMAGE_URI" \
  -var "infolang_api_key_secret_arn=$SECRET_ARN"
```

See `main.tf` for the ECR repo, execution role (trust:
`bedrock-agentcore.amazonaws.com`), and the `awscc_bedrockagentcore_runtime`
resource. Verify the AgentCore resource schema against your installed `awscc`
provider version before applying — Cloud Control schemas track the API and
change over time.

## Verifying durability across a session recreate

```bash
SID1=$(uuidgen)
curl -s "$RUNTIME_URL/invocations" -H "X-Amzn-Bedrock-AgentCore-Runtime-Session-Id: $SID1" \
  -d '{"prompt":"Remember our SLA is 99.9%","actor_id":"customer-42"}'

# New session id == as if the runtime was torn down and recreated.
SID2=$(uuidgen)
curl -s "$RUNTIME_URL/invocations" -H "X-Amzn-Bedrock-AgentCore-Runtime-Session-Id: $SID2" \
  -d '{"prompt":"What is our SLA?","actor_id":"customer-42"}'
# -> recalls the 99.9% SLA, because memory is keyed by actor_id, not the session.
```
