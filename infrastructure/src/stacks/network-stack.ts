import * as cdk from 'aws-cdk-lib';
import * as ssm from 'aws-cdk-lib/aws-ssm';
import { Construct } from 'constructs';
import { VoiceEngineConfig } from '../config';
import { ssmParams } from '../ssm-parameters';
import { VpcConstruct } from '../constructs';

export interface NetworkStackProps extends cdk.StackProps {
  readonly config: VoiceEngineConfig;
}

/**
 * Network stack — owns the VPC and exports its IDs via SSM so other
 * stacks can look them up without creating a CFN cyclic dependency.
 *
 * v1's NetworkStack used `Vpc.fromLookup` in downstream stacks; v2 does
 * the same. The looked-up VPC's CIDR + AZ count is resolved at synth
 * time, so changes to the VPC require a fresh synth of all dependent
 * stacks.
 */
export class NetworkStack extends cdk.Stack {
  public readonly vpcConstruct: VpcConstruct;

  constructor(scope: Construct, id: string, props: NetworkStackProps) {
    super(scope, id, props);
    const { config } = props;

    this.vpcConstruct = new VpcConstruct(this, 'Vpc', { config });
    const params = ssmParams(config);

    new ssm.StringParameter(this, 'VpcIdParam', {
      parameterName: params.VPC_ID,
      stringValue: this.vpcConstruct.vpc.vpcId,
      description: 'VPC ID for the voice engine.',
    });
    new ssm.StringListParameter(this, 'PrivateSubnetIdsParam', {
      parameterName: params.PRIVATE_SUBNET_IDS,
      stringListValue: this.vpcConstruct.taskSubnetIds,
      description: 'Private subnet IDs for Fargate tasks.',
    });
    new ssm.StringListParameter(this, 'PublicSubnetIdsParam', {
      parameterName: params.PUBLIC_SUBNET_IDS,
      stringListValue: this.vpcConstruct.albSubnetIds,
      description: 'Public subnet IDs for the ALB.',
    });
    new ssm.StringParameter(this, 'TaskSecurityGroupIdParam', {
      parameterName: params.TASK_SECURITY_GROUP_ID,
      stringValue: this.vpcConstruct.taskSecurityGroup.securityGroupId,
      description: 'Security group ID applied to Fargate tasks.',
    });
    new ssm.StringParameter(this, 'AlbSecurityGroupIdParam', {
      parameterName: params.ALB_SECURITY_GROUP_ID,
      stringValue: this.vpcConstruct.albSecurityGroup.securityGroupId,
      description: 'Security group ID applied to the ALB.',
    });

    new cdk.CfnOutput(this, 'VpcId', { value: this.vpcConstruct.vpc.vpcId });
    new cdk.CfnOutput(this, 'NatGatewayCount', { value: String(config.natGateways) });
  }
}
