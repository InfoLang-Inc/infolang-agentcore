# WP39 x WP17: deploy the InfoLang-backed AgentCore agent into a customer's
# dedicated AWS account.
#
# This provisions the supporting infra with the well-known AWS provider
# (ECR + IAM execution role) and the AgentCore Runtime itself with the AWS
# Cloud Control (awscc) provider, which tracks the CloudFormation resource
# `AWS::BedrockAgentCore::Runtime`.
#
# NOTE: AgentCore resource schemas evolve. Verify `awscc_bedrockagentcore_runtime`
# and its argument names against your installed awscc provider version
# (`terraform providers schema -json`) before applying. The starter toolkit
# (`agentcore configure && agentcore launch`) remains the quickest path; use
# this Terraform when you need the per-customer-account, IaC-managed model.

terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = ">= 5.60"
    }
    awscc = {
      source  = "hashicorp/awscc"
      version = ">= 1.0"
    }
  }
}

variable "region" {
  type    = string
  default = "us-west-2"
}

variable "agent_name" {
  type    = string
  default = "infolang-agentcore-agent"
}

variable "image_uri" {
  type        = string
  description = "linux/arm64 image URI in ECR (built from examples/Dockerfile)."
}

variable "infolang_api_key_secret_arn" {
  type        = string
  description = "Secrets Manager ARN holding the InfoLang API key."
}

provider "aws" {
  region = var.region
}

provider "awscc" {
  region = var.region
}

# --- container registry ---------------------------------------------------

resource "aws_ecr_repository" "agent" {
  name                 = var.agent_name
  image_tag_mutability = "IMMUTABLE"
  image_scanning_configuration {
    scan_on_push = true
  }
}

# --- execution role -------------------------------------------------------

data "aws_iam_policy_document" "assume" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["bedrock-agentcore.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "runtime" {
  name               = "${var.agent_name}-exec"
  assume_role_policy = data.aws_iam_policy_document.assume.json
}

data "aws_iam_policy_document" "runtime" {
  # Pull the agent image.
  statement {
    actions = [
      "ecr:GetDownloadUrlForLayer",
      "ecr:BatchGetImage",
      "ecr:GetAuthorizationToken",
    ]
    resources = ["*"]
  }
  # Read the InfoLang API key at runtime.
  statement {
    actions   = ["secretsmanager:GetSecretValue"]
    resources = [var.infolang_api_key_secret_arn]
  }
  # Ship logs.
  statement {
    actions = [
      "logs:CreateLogGroup",
      "logs:CreateLogStream",
      "logs:PutLogEvents",
    ]
    resources = ["*"]
  }
}

resource "aws_iam_role_policy" "runtime" {
  name   = "${var.agent_name}-policy"
  role   = aws_iam_role.runtime.id
  policy = data.aws_iam_policy_document.runtime.json
}

# --- AgentCore Runtime ----------------------------------------------------

resource "awscc_bedrockagentcore_runtime" "agent" {
  agent_runtime_name = replace(var.agent_name, "-", "_")
  role_arn           = aws_iam_role.runtime.arn

  agent_runtime_artifact = {
    container_configuration = {
      container_uri = var.image_uri
    }
  }

  network_configuration = {
    network_mode = "PUBLIC"
  }

  environment_variables = {
    INFOLANG_API_KEY_SECRET_ARN = var.infolang_api_key_secret_arn
  }
}

output "agent_runtime_arn" {
  value = awscc_bedrockagentcore_runtime.agent.agent_runtime_arn
}
