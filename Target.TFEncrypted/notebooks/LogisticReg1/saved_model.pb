��
��
8
Const
output"dtype"
valuetensor"
dtypetype

NoOp
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype�
�
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring �
q
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape�"serve*1.15.52v1.15.4-39-g3db52be8��
�
onnx__Sigmoid_3_5/kernelVarHandleOp*
dtype0*)
shared_nameonnx__Sigmoid_3_5/kernel*
_output_shapes
: *
shape
:
�
,onnx__Sigmoid_3_5/kernel/Read/ReadVariableOpReadVariableOponnx__Sigmoid_3_5/kernel*
_output_shapes

:*
dtype0
�
onnx__Sigmoid_3_5/biasVarHandleOp*'
shared_nameonnx__Sigmoid_3_5/bias*
shape:*
dtype0*
_output_shapes
: 
}
*onnx__Sigmoid_3_5/bias/Read/ReadVariableOpReadVariableOponnx__Sigmoid_3_5/bias*
_output_shapes
:*
dtype0

NoOpNoOp
�
ConstConst"/device:CPU:0*
_output_shapes
: *�
value�B� B�
�
layer-0
layer_with_weights-0
layer-1
layer-2
trainable_variables
	variables
regularization_losses
	keras_api

signatures
R
	trainable_variables

	variables
regularization_losses
	keras_api
~

kernel
bias
_callable_losses
trainable_variables
	variables
regularization_losses
	keras_api
h
_callable_losses
trainable_variables
	variables
regularization_losses
	keras_api

0
1

0
1
 
�
trainable_variables
metrics
	variables

layers
regularization_losses
layer_regularization_losses
non_trainable_variables
 
 
 
 
�
	trainable_variables
metrics

layers

	variables
regularization_losses
layer_regularization_losses
 non_trainable_variables
db
VARIABLE_VALUEonnx__Sigmoid_3_5/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
`^
VARIABLE_VALUEonnx__Sigmoid_3_5/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
 
�
trainable_variables
!metrics

"layers
	variables
regularization_losses
#layer_regularization_losses
$non_trainable_variables
 
 
 
 
�
trainable_variables
%metrics

&layers
	variables
regularization_losses
'layer_regularization_losses
(non_trainable_variables
 

0
1
2
 
 
 
 
 
 
 
 
 
 
 
 
 
 *
dtype0

serving_default_onnx__Gemm_0Placeholder*
shape:���������*'
_output_shapes
:���������*
dtype0
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_onnx__Gemm_0onnx__Sigmoid_3_5/kernelonnx__Sigmoid_3_5/bias*'
_output_shapes
:���������*
Tout
2*+
f&R$
"__inference_signature_wrapper_1075*
Tin
2*+
_gradient_op_typePartitionedCall-1149**
config_proto

CPU

GPU 2J 8
O
saver_filenamePlaceholder*
_output_shapes
: *
shape: *
dtype0
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename,onnx__Sigmoid_3_5/kernel/Read/ReadVariableOp*onnx__Sigmoid_3_5/bias/Read/ReadVariableOpConst*&
f!R
__inference__traced_save_1172*
Tin
2*
Tout
2*+
_gradient_op_typePartitionedCall-1173**
config_proto

CPU

GPU 2J 8*
_output_shapes
: 
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameonnx__Sigmoid_3_5/kernelonnx__Sigmoid_3_5/bias**
config_proto

CPU

GPU 2J 8*+
_gradient_op_typePartitionedCall-1192*
Tout
2*)
f$R"
 __inference__traced_restore_1191*
_output_shapes
: *
Tin
2��
�
�
__inference__wrapped_model_962
onnx__gemm_0J
Fmodel_5_onnx__sigmoid_3_matmul_readvariableop_onnx__sigmoid_3_5_kernelI
Emodel_5_onnx__sigmoid_3_biasadd_readvariableop_onnx__sigmoid_3_5_bias
identity��.model_5/onnx__Sigmoid_3/BiasAdd/ReadVariableOp�-model_5/onnx__Sigmoid_3/MatMul/ReadVariableOp�
-model_5/onnx__Sigmoid_3/MatMul/ReadVariableOpReadVariableOpFmodel_5_onnx__sigmoid_3_matmul_readvariableop_onnx__sigmoid_3_5_kernel*
_output_shapes

:*
dtype0�
model_5/onnx__Sigmoid_3/MatMulMatMulonnx__gemm_05model_5/onnx__Sigmoid_3/MatMul/ReadVariableOp:value:0*'
_output_shapes
:���������*
T0�
.model_5/onnx__Sigmoid_3/BiasAdd/ReadVariableOpReadVariableOpEmodel_5_onnx__sigmoid_3_biasadd_readvariableop_onnx__sigmoid_3_5_bias*
dtype0*
_output_shapes
:�
model_5/onnx__Sigmoid_3/BiasAddBiasAdd(model_5/onnx__Sigmoid_3/MatMul:product:06model_5/onnx__Sigmoid_3/BiasAdd/ReadVariableOp:value:0*'
_output_shapes
:���������*
T0x
model_5/4/SigmoidSigmoid(model_5/onnx__Sigmoid_3/BiasAdd:output:0*'
_output_shapes
:���������*
T0�
IdentityIdentitymodel_5/4/Sigmoid:y:0/^model_5/onnx__Sigmoid_3/BiasAdd/ReadVariableOp.^model_5/onnx__Sigmoid_3/MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������"
identityIdentity:output:0*.
_input_shapes
:���������::2`
.model_5/onnx__Sigmoid_3/BiasAdd/ReadVariableOp.model_5/onnx__Sigmoid_3/BiasAdd/ReadVariableOp2^
-model_5/onnx__Sigmoid_3/MatMul/ReadVariableOp-model_5/onnx__Sigmoid_3/MatMul/ReadVariableOp:, (
&
_user_specified_nameonnx__Gemm_0: : 
�

�
A__inference_model_5_layer_call_and_return_conditional_losses_1041

inputsD
@onnx__sigmoid_3_statefulpartitionedcall_onnx__sigmoid_3_5_kernelB
>onnx__sigmoid_3_statefulpartitionedcall_onnx__sigmoid_3_5_bias
identity��'onnx__Sigmoid_3/StatefulPartitionedCall�
'onnx__Sigmoid_3/StatefulPartitionedCallStatefulPartitionedCallinputs@onnx__sigmoid_3_statefulpartitionedcall_onnx__sigmoid_3_5_kernel>onnx__sigmoid_3_statefulpartitionedcall_onnx__sigmoid_3_5_bias**
_gradient_op_typePartitionedCall-986*'
_output_shapes
:���������**
config_proto

CPU

GPU 2J 8*Q
fLRJ
H__inference_onnx__Sigmoid_3_layer_call_and_return_conditional_losses_979*
Tout
2*
Tin
2�
4/PartitionedCallPartitionedCall0onnx__Sigmoid_3/StatefulPartitionedCall:output:0*
Tout
2**
config_proto

CPU

GPU 2J 8*+
_gradient_op_typePartitionedCall-1010*D
f?R=
;__inference_4_layer_call_and_return_conditional_losses_1003*'
_output_shapes
:���������*
Tin
2�
IdentityIdentity4/PartitionedCall:output:0(^onnx__Sigmoid_3/StatefulPartitionedCall*
T0*'
_output_shapes
:���������"
identityIdentity:output:0*.
_input_shapes
:���������::2R
'onnx__Sigmoid_3/StatefulPartitionedCall'onnx__Sigmoid_3/StatefulPartitionedCall: :& "
 
_user_specified_nameinputs: 
�
�
A__inference_model_5_layer_call_and_return_conditional_losses_1089

inputsB
>onnx__sigmoid_3_matmul_readvariableop_onnx__sigmoid_3_5_kernelA
=onnx__sigmoid_3_biasadd_readvariableop_onnx__sigmoid_3_5_bias
identity��&onnx__Sigmoid_3/BiasAdd/ReadVariableOp�%onnx__Sigmoid_3/MatMul/ReadVariableOp�
%onnx__Sigmoid_3/MatMul/ReadVariableOpReadVariableOp>onnx__sigmoid_3_matmul_readvariableop_onnx__sigmoid_3_5_kernel*
_output_shapes

:*
dtype0�
onnx__Sigmoid_3/MatMulMatMulinputs-onnx__Sigmoid_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
&onnx__Sigmoid_3/BiasAdd/ReadVariableOpReadVariableOp=onnx__sigmoid_3_biasadd_readvariableop_onnx__sigmoid_3_5_bias*
dtype0*
_output_shapes
:�
onnx__Sigmoid_3/BiasAddBiasAdd onnx__Sigmoid_3/MatMul:product:0.onnx__Sigmoid_3/BiasAdd/ReadVariableOp:value:0*'
_output_shapes
:���������*
T0h
	4/SigmoidSigmoid onnx__Sigmoid_3/BiasAdd:output:0*
T0*'
_output_shapes
:����������
IdentityIdentity4/Sigmoid:y:0'^onnx__Sigmoid_3/BiasAdd/ReadVariableOp&^onnx__Sigmoid_3/MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������"
identityIdentity:output:0*.
_input_shapes
:���������::2N
%onnx__Sigmoid_3/MatMul/ReadVariableOp%onnx__Sigmoid_3/MatMul/ReadVariableOp2P
&onnx__Sigmoid_3/BiasAdd/ReadVariableOp&onnx__Sigmoid_3/BiasAdd/ReadVariableOp:& "
 
_user_specified_nameinputs: : 
�
�
.__inference_onnx__Sigmoid_3_layer_call_fn_1131

inputs4
0statefulpartitionedcall_onnx__sigmoid_3_5_kernel2
.statefulpartitionedcall_onnx__sigmoid_3_5_bias
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs0statefulpartitionedcall_onnx__sigmoid_3_5_kernel.statefulpartitionedcall_onnx__sigmoid_3_5_bias*
Tin
2**
config_proto

CPU

GPU 2J 8*
Tout
2*Q
fLRJ
H__inference_onnx__Sigmoid_3_layer_call_and_return_conditional_losses_979*'
_output_shapes
:���������**
_gradient_op_typePartitionedCall-986�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������"
identityIdentity:output:0*.
_input_shapes
:���������::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs: : 
�
W
;__inference_4_layer_call_and_return_conditional_losses_1136

inputs
identityL
SigmoidSigmoidinputs*'
_output_shapes
:���������*
T0S
IdentityIdentitySigmoid:y:0*
T0*'
_output_shapes
:���������"
identityIdentity:output:0*&
_input_shapes
:���������:& "
 
_user_specified_nameinputs
�
W
;__inference_4_layer_call_and_return_conditional_losses_1003

inputs
identityL
SigmoidSigmoidinputs*'
_output_shapes
:���������*
T0S
IdentityIdentitySigmoid:y:0*
T0*'
_output_shapes
:���������"
identityIdentity:output:0*&
_input_shapes
:���������:& "
 
_user_specified_nameinputs
�
�
&__inference_model_5_layer_call_fn_1107

inputs4
0statefulpartitionedcall_onnx__sigmoid_3_5_kernel2
.statefulpartitionedcall_onnx__sigmoid_3_5_bias
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs0statefulpartitionedcall_onnx__sigmoid_3_5_kernel.statefulpartitionedcall_onnx__sigmoid_3_5_bias*'
_output_shapes
:���������*J
fERC
A__inference_model_5_layer_call_and_return_conditional_losses_1041*
Tout
2*
Tin
2**
config_proto

CPU

GPU 2J 8*+
_gradient_op_typePartitionedCall-1042�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������"
identityIdentity:output:0*.
_input_shapes
:���������::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs: : 
�
�
&__inference_model_5_layer_call_fn_1047
onnx__gemm_04
0statefulpartitionedcall_onnx__sigmoid_3_5_kernel2
.statefulpartitionedcall_onnx__sigmoid_3_5_bias
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallonnx__gemm_00statefulpartitionedcall_onnx__sigmoid_3_5_kernel.statefulpartitionedcall_onnx__sigmoid_3_5_bias*+
_gradient_op_typePartitionedCall-1042*
Tout
2*J
fERC
A__inference_model_5_layer_call_and_return_conditional_losses_1041**
config_proto

CPU

GPU 2J 8*
Tin
2*'
_output_shapes
:����������
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*'
_output_shapes
:���������*
T0"
identityIdentity:output:0*.
_input_shapes
:���������::22
StatefulPartitionedCallStatefulPartitionedCall:, (
&
_user_specified_nameonnx__Gemm_0: : 
�
�
A__inference_model_5_layer_call_and_return_conditional_losses_1019
onnx__gemm_0D
@onnx__sigmoid_3_statefulpartitionedcall_onnx__sigmoid_3_5_kernelB
>onnx__sigmoid_3_statefulpartitionedcall_onnx__sigmoid_3_5_bias
identity��'onnx__Sigmoid_3/StatefulPartitionedCall�
'onnx__Sigmoid_3/StatefulPartitionedCallStatefulPartitionedCallonnx__gemm_0@onnx__sigmoid_3_statefulpartitionedcall_onnx__sigmoid_3_5_kernel>onnx__sigmoid_3_statefulpartitionedcall_onnx__sigmoid_3_5_bias**
_gradient_op_typePartitionedCall-986*'
_output_shapes
:���������*
Tin
2*
Tout
2**
config_proto

CPU

GPU 2J 8*Q
fLRJ
H__inference_onnx__Sigmoid_3_layer_call_and_return_conditional_losses_979�
4/PartitionedCallPartitionedCall0onnx__Sigmoid_3/StatefulPartitionedCall:output:0*D
f?R=
;__inference_4_layer_call_and_return_conditional_losses_1003**
config_proto

CPU

GPU 2J 8*
Tout
2*+
_gradient_op_typePartitionedCall-1010*'
_output_shapes
:���������*
Tin
2�
IdentityIdentity4/PartitionedCall:output:0(^onnx__Sigmoid_3/StatefulPartitionedCall*'
_output_shapes
:���������*
T0"
identityIdentity:output:0*.
_input_shapes
:���������::2R
'onnx__Sigmoid_3/StatefulPartitionedCall'onnx__Sigmoid_3/StatefulPartitionedCall:, (
&
_user_specified_nameonnx__Gemm_0: : 
�
�
I__inference_onnx__Sigmoid_3_layer_call_and_return_conditional_losses_1124

inputs2
.matmul_readvariableop_onnx__sigmoid_3_5_kernel1
-biasadd_readvariableop_onnx__sigmoid_3_5_bias
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOp.matmul_readvariableop_onnx__sigmoid_3_5_kernel*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
BiasAdd/ReadVariableOpReadVariableOp-biasadd_readvariableop_onnx__sigmoid_3_5_bias*
dtype0*
_output_shapes
:v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*'
_output_shapes
:���������*
T0�
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*'
_output_shapes
:���������*
T0"
identityIdentity:output:0*.
_input_shapes
:���������::2.
MatMul/ReadVariableOpMatMul/ReadVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp:& "
 
_user_specified_nameinputs: : 
�
�
 __inference__traced_restore_1191
file_prefix-
)assignvariableop_onnx__sigmoid_3_5_kernel-
)assignvariableop_1_onnx__sigmoid_3_5_bias

identity_3��AssignVariableOp�AssignVariableOp_1�	RestoreV2�RestoreV2_1�
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�
valuexBvB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEt
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
valueBB B *
dtype0�
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*
dtypes
2*
_output_shapes

::L
IdentityIdentityRestoreV2:tensors:0*
T0*
_output_shapes
:�
AssignVariableOpAssignVariableOp)assignvariableop_onnx__sigmoid_3_5_kernelIdentity:output:0*
dtype0*
_output_shapes
 N

Identity_1IdentityRestoreV2:tensors:1*
T0*
_output_shapes
:�
AssignVariableOp_1AssignVariableOp)assignvariableop_1_onnx__sigmoid_3_5_biasIdentity_1:output:0*
dtype0*
_output_shapes
 �
RestoreV2_1/tensor_namesConst"/device:CPU:0*
dtype0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH*
_output_shapes
:t
RestoreV2_1/shape_and_slicesConst"/device:CPU:0*
valueB
B *
_output_shapes
:*
dtype0�
RestoreV2_1	RestoreV2file_prefix!RestoreV2_1/tensor_names:output:0%RestoreV2_1/shape_and_slices:output:0
^RestoreV2"/device:CPU:0*
dtypes
2*
_output_shapes
:1
NoOpNoOp"/device:CPU:0*
_output_shapes
 �

Identity_2Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^NoOp"/device:CPU:0*
_output_shapes
: *
T0�

Identity_3IdentityIdentity_2:output:0^AssignVariableOp^AssignVariableOp_1
^RestoreV2^RestoreV2_1*
_output_shapes
: *
T0"!

identity_3Identity_3:output:0*
_input_shapes

: ::2(
AssignVariableOp_1AssignVariableOp_12
RestoreV2_1RestoreV2_12$
AssignVariableOpAssignVariableOp2
	RestoreV2	RestoreV2: :+ '
%
_user_specified_namefile_prefix: 
�
�
A__inference_model_5_layer_call_and_return_conditional_losses_1100

inputsB
>onnx__sigmoid_3_matmul_readvariableop_onnx__sigmoid_3_5_kernelA
=onnx__sigmoid_3_biasadd_readvariableop_onnx__sigmoid_3_5_bias
identity��&onnx__Sigmoid_3/BiasAdd/ReadVariableOp�%onnx__Sigmoid_3/MatMul/ReadVariableOp�
%onnx__Sigmoid_3/MatMul/ReadVariableOpReadVariableOp>onnx__sigmoid_3_matmul_readvariableop_onnx__sigmoid_3_5_kernel*
dtype0*
_output_shapes

:�
onnx__Sigmoid_3/MatMulMatMulinputs-onnx__Sigmoid_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
&onnx__Sigmoid_3/BiasAdd/ReadVariableOpReadVariableOp=onnx__sigmoid_3_biasadd_readvariableop_onnx__sigmoid_3_5_bias*
_output_shapes
:*
dtype0�
onnx__Sigmoid_3/BiasAddBiasAdd onnx__Sigmoid_3/MatMul:product:0.onnx__Sigmoid_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������h
	4/SigmoidSigmoid onnx__Sigmoid_3/BiasAdd:output:0*'
_output_shapes
:���������*
T0�
IdentityIdentity4/Sigmoid:y:0'^onnx__Sigmoid_3/BiasAdd/ReadVariableOp&^onnx__Sigmoid_3/MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������"
identityIdentity:output:0*.
_input_shapes
:���������::2P
&onnx__Sigmoid_3/BiasAdd/ReadVariableOp&onnx__Sigmoid_3/BiasAdd/ReadVariableOp2N
%onnx__Sigmoid_3/MatMul/ReadVariableOp%onnx__Sigmoid_3/MatMul/ReadVariableOp:& "
 
_user_specified_nameinputs: : 
�
�
A__inference_model_5_layer_call_and_return_conditional_losses_1030
onnx__gemm_0D
@onnx__sigmoid_3_statefulpartitionedcall_onnx__sigmoid_3_5_kernelB
>onnx__sigmoid_3_statefulpartitionedcall_onnx__sigmoid_3_5_bias
identity��'onnx__Sigmoid_3/StatefulPartitionedCall�
'onnx__Sigmoid_3/StatefulPartitionedCallStatefulPartitionedCallonnx__gemm_0@onnx__sigmoid_3_statefulpartitionedcall_onnx__sigmoid_3_5_kernel>onnx__sigmoid_3_statefulpartitionedcall_onnx__sigmoid_3_5_bias**
_gradient_op_typePartitionedCall-986*'
_output_shapes
:���������**
config_proto

CPU

GPU 2J 8*Q
fLRJ
H__inference_onnx__Sigmoid_3_layer_call_and_return_conditional_losses_979*
Tin
2*
Tout
2�
4/PartitionedCallPartitionedCall0onnx__Sigmoid_3/StatefulPartitionedCall:output:0*
Tout
2*+
_gradient_op_typePartitionedCall-1010*D
f?R=
;__inference_4_layer_call_and_return_conditional_losses_1003**
config_proto

CPU

GPU 2J 8*
Tin
2*'
_output_shapes
:����������
IdentityIdentity4/PartitionedCall:output:0(^onnx__Sigmoid_3/StatefulPartitionedCall*'
_output_shapes
:���������*
T0"
identityIdentity:output:0*.
_input_shapes
:���������::2R
'onnx__Sigmoid_3/StatefulPartitionedCall'onnx__Sigmoid_3/StatefulPartitionedCall:, (
&
_user_specified_nameonnx__Gemm_0: : 
�
�
&__inference_model_5_layer_call_fn_1114

inputs4
0statefulpartitionedcall_onnx__sigmoid_3_5_kernel2
.statefulpartitionedcall_onnx__sigmoid_3_5_bias
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs0statefulpartitionedcall_onnx__sigmoid_3_5_kernel.statefulpartitionedcall_onnx__sigmoid_3_5_bias*J
fERC
A__inference_model_5_layer_call_and_return_conditional_losses_1060*
Tin
2*+
_gradient_op_typePartitionedCall-1061*
Tout
2**
config_proto

CPU

GPU 2J 8*'
_output_shapes
:����������
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������"
identityIdentity:output:0*.
_input_shapes
:���������::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs: : 
�
�
"__inference_signature_wrapper_1075
onnx__gemm_04
0statefulpartitionedcall_onnx__sigmoid_3_5_kernel2
.statefulpartitionedcall_onnx__sigmoid_3_5_bias
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallonnx__gemm_00statefulpartitionedcall_onnx__sigmoid_3_5_kernel.statefulpartitionedcall_onnx__sigmoid_3_5_bias*+
_gradient_op_typePartitionedCall-1070*'
_output_shapes
:���������**
config_proto

CPU

GPU 2J 8*
Tin
2*'
f"R 
__inference__wrapped_model_962*
Tout
2�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*'
_output_shapes
:���������*
T0"
identityIdentity:output:0*.
_input_shapes
:���������::22
StatefulPartitionedCallStatefulPartitionedCall:, (
&
_user_specified_nameonnx__Gemm_0: : 
�

�
A__inference_model_5_layer_call_and_return_conditional_losses_1060

inputsD
@onnx__sigmoid_3_statefulpartitionedcall_onnx__sigmoid_3_5_kernelB
>onnx__sigmoid_3_statefulpartitionedcall_onnx__sigmoid_3_5_bias
identity��'onnx__Sigmoid_3/StatefulPartitionedCall�
'onnx__Sigmoid_3/StatefulPartitionedCallStatefulPartitionedCallinputs@onnx__sigmoid_3_statefulpartitionedcall_onnx__sigmoid_3_5_kernel>onnx__sigmoid_3_statefulpartitionedcall_onnx__sigmoid_3_5_bias*Q
fLRJ
H__inference_onnx__Sigmoid_3_layer_call_and_return_conditional_losses_979**
_gradient_op_typePartitionedCall-986*
Tin
2**
config_proto

CPU

GPU 2J 8*'
_output_shapes
:���������*
Tout
2�
4/PartitionedCallPartitionedCall0onnx__Sigmoid_3/StatefulPartitionedCall:output:0*+
_gradient_op_typePartitionedCall-1010*'
_output_shapes
:���������**
config_proto

CPU

GPU 2J 8*D
f?R=
;__inference_4_layer_call_and_return_conditional_losses_1003*
Tout
2*
Tin
2�
IdentityIdentity4/PartitionedCall:output:0(^onnx__Sigmoid_3/StatefulPartitionedCall*
T0*'
_output_shapes
:���������"
identityIdentity:output:0*.
_input_shapes
:���������::2R
'onnx__Sigmoid_3/StatefulPartitionedCall'onnx__Sigmoid_3/StatefulPartitionedCall:& "
 
_user_specified_nameinputs: : 
�
�
H__inference_onnx__Sigmoid_3_layer_call_and_return_conditional_losses_979

inputs2
.matmul_readvariableop_onnx__sigmoid_3_5_kernel1
-biasadd_readvariableop_onnx__sigmoid_3_5_bias
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOp.matmul_readvariableop_onnx__sigmoid_3_5_kernel*
dtype0*
_output_shapes

:i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*'
_output_shapes
:���������*
T0�
BiasAdd/ReadVariableOpReadVariableOp-biasadd_readvariableop_onnx__sigmoid_3_5_bias*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������"
identityIdentity:output:0*.
_input_shapes
:���������::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:& "
 
_user_specified_nameinputs: : 
�
�
__inference__traced_save_1172
file_prefix7
3savev2_onnx__sigmoid_3_5_kernel_read_readvariableop5
1savev2_onnx__sigmoid_3_5_bias_read_readvariableop
savev2_1_const

identity_1��MergeV2Checkpoints�SaveV2�SaveV2_1�
StringJoin/inputs_1Const"/device:CPU:0*
dtype0*
_output_shapes
: *<
value3B1 B+_temp_64cae81c35504794bf047dd7aa8ee772/parts

StringJoin
StringJoinfile_prefixStringJoin/inputs_1:output:0"/device:CPU:0*
_output_shapes
: *
NL

num_shardsConst*
value	B :*
_output_shapes
: *
dtype0f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
value	B : *
dtype0�
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: �
SaveV2/tensor_namesConst"/device:CPU:0*�
valuexBvB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:q
SaveV2/shape_and_slicesConst"/device:CPU:0*
dtype0*
_output_shapes
:*
valueBB B �
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:03savev2_onnx__sigmoid_3_5_kernel_read_readvariableop1savev2_onnx__sigmoid_3_5_bias_read_readvariableop"/device:CPU:0*
dtypes
2*
_output_shapes
 h
ShardedFilename_1/shardConst"/device:CPU:0*
value	B :*
_output_shapes
: *
dtype0�
ShardedFilename_1ShardedFilenameStringJoin:output:0 ShardedFilename_1/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: �
SaveV2_1/tensor_namesConst"/device:CPU:0*
dtype0*
_output_shapes
:*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPHq
SaveV2_1/shape_and_slicesConst"/device:CPU:0*
valueB
B *
_output_shapes
:*
dtype0�
SaveV2_1SaveV2ShardedFilename_1:filename:0SaveV2_1/tensor_names:output:0"SaveV2_1/shape_and_slices:output:0savev2_1_const^SaveV2"/device:CPU:0*
_output_shapes
 *
dtypes
2�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0ShardedFilename_1:filename:0^SaveV2	^SaveV2_1"/device:CPU:0*
_output_shapes
:*
T0*
N�
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix	^SaveV2_1"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: s

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints^SaveV2	^SaveV2_1*
_output_shapes
: *
T0"!

identity_1Identity_1:output:0*'
_input_shapes
: ::: 2
SaveV2_1SaveV2_12(
MergeV2CheckpointsMergeV2Checkpoints2
SaveV2SaveV2:+ '
%
_user_specified_namefile_prefix: : : 
�
<
 __inference_4_layer_call_fn_1141

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*'
_output_shapes
:���������**
config_proto

CPU

GPU 2J 8*D
f?R=
;__inference_4_layer_call_and_return_conditional_losses_1003*+
_gradient_op_typePartitionedCall-1010*
Tout
2`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:���������"
identityIdentity:output:0*&
_input_shapes
:���������:& "
 
_user_specified_nameinputs
�
�
&__inference_model_5_layer_call_fn_1066
onnx__gemm_04
0statefulpartitionedcall_onnx__sigmoid_3_5_kernel2
.statefulpartitionedcall_onnx__sigmoid_3_5_bias
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallonnx__gemm_00statefulpartitionedcall_onnx__sigmoid_3_5_kernel.statefulpartitionedcall_onnx__sigmoid_3_5_bias*'
_output_shapes
:���������*
Tout
2*+
_gradient_op_typePartitionedCall-1061*J
fERC
A__inference_model_5_layer_call_and_return_conditional_losses_1060**
config_proto

CPU

GPU 2J 8*
Tin
2�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*'
_output_shapes
:���������*
T0"
identityIdentity:output:0*.
_input_shapes
:���������::22
StatefulPartitionedCallStatefulPartitionedCall: :, (
&
_user_specified_nameonnx__Gemm_0: "�L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
E
onnx__Gemm_05
serving_default_onnx__Gemm_0:0���������5
40
StatefulPartitionedCall:0���������tensorflow/serving/predict:�Y
�
layer-0
layer_with_weights-0
layer-1
layer-2
trainable_variables
	variables
regularization_losses
	keras_api

signatures
)__call__
**&call_and_return_all_conditional_losses
+_default_save_signature"�
_tf_keras_model�{"class_name": "Model", "name": "model_5", "trainable": true, "expects_training_arg": true, "dtype": null, "batch_input_shape": null, "config": {"name": "model_5", "layers": [{"name": "onnx__Gemm_0", "class_name": "InputLayer", "config": {"batch_input_shape": [null, 2], "dtype": "float32", "sparse": false, "ragged": false, "name": "onnx__Gemm_0"}, "inbound_nodes": []}, {"name": "onnx__Sigmoid_3", "class_name": "Dense", "config": {"name": "onnx__Sigmoid_3", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "Zeros", "config": {"dtype": "float32"}}, "bias_initializer": {"class_name": "Zeros", "config": {"dtype": "float32"}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["onnx__Gemm_0", 0, 0, {}]]]}, {"name": "4", "class_name": "Activation", "config": {"name": "4", "trainable": true, "dtype": "float32", "activation": "sigmoid"}, "inbound_nodes": [[["onnx__Sigmoid_3", 0, 0, {}]]]}], "input_layers": [["onnx__Gemm_0", 0, 0]], "output_layers": [["4", 0, 0]]}, "input_spec": null, "activity_regularizer": null, "keras_version": "2.2.4-tf", "backend": "tensorflow", "model_config": {"class_name": "Model", "config": {"name": "model_5", "layers": [{"name": "onnx__Gemm_0", "class_name": "InputLayer", "config": {"batch_input_shape": [null, 2], "dtype": "float32", "sparse": false, "ragged": false, "name": "onnx__Gemm_0"}, "inbound_nodes": []}, {"name": "onnx__Sigmoid_3", "class_name": "Dense", "config": {"name": "onnx__Sigmoid_3", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "Zeros", "config": {"dtype": "float32"}}, "bias_initializer": {"class_name": "Zeros", "config": {"dtype": "float32"}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["onnx__Gemm_0", 0, 0, {}]]]}, {"name": "4", "class_name": "Activation", "config": {"name": "4", "trainable": true, "dtype": "float32", "activation": "sigmoid"}, "inbound_nodes": [[["onnx__Sigmoid_3", 0, 0, {}]]]}], "input_layers": [["onnx__Gemm_0", 0, 0]], "output_layers": [["4", 0, 0]]}}}
�
	trainable_variables

	variables
regularization_losses
	keras_api
,__call__
*-&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "InputLayer", "name": "onnx__Gemm_0", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": [null, 2], "config": {"batch_input_shape": [null, 2], "dtype": "float32", "sparse": false, "ragged": false, "name": "onnx__Gemm_0"}, "input_spec": null, "activity_regularizer": null}
�

kernel
bias
_callable_losses
trainable_variables
	variables
regularization_losses
	keras_api
.__call__
*/&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "Dense", "name": "onnx__Sigmoid_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "onnx__Sigmoid_3", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "Zeros", "config": {"dtype": "float32"}}, "bias_initializer": {"class_name": "Zeros", "config": {"dtype": "float32"}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 2}}}, "activity_regularizer": null}
�
_callable_losses
trainable_variables
	variables
regularization_losses
	keras_api
0__call__
*1&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "Activation", "name": "4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "4", "trainable": true, "dtype": "float32", "activation": "sigmoid"}, "input_spec": null, "activity_regularizer": null}
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
trainable_variables
metrics
	variables

layers
regularization_losses
layer_regularization_losses
non_trainable_variables
)__call__
+_default_save_signature
**&call_and_return_all_conditional_losses
&*"call_and_return_conditional_losses"
_generic_user_object
,
2serving_default"
signature_map
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
	trainable_variables
metrics

layers

	variables
regularization_losses
layer_regularization_losses
 non_trainable_variables
,__call__
*-&call_and_return_all_conditional_losses
&-"call_and_return_conditional_losses"
_generic_user_object
*:(2onnx__Sigmoid_3_5/kernel
$:"2onnx__Sigmoid_3_5/bias
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
trainable_variables
!metrics

"layers
	variables
regularization_losses
#layer_regularization_losses
$non_trainable_variables
.__call__
*/&call_and_return_all_conditional_losses
&/"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
trainable_variables
%metrics

&layers
	variables
regularization_losses
'layer_regularization_losses
(non_trainable_variables
0__call__
*1&call_and_return_all_conditional_losses
&1"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
5
0
1
2"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�2�
&__inference_model_5_layer_call_fn_1047
&__inference_model_5_layer_call_fn_1107
&__inference_model_5_layer_call_fn_1114
&__inference_model_5_layer_call_fn_1066�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
A__inference_model_5_layer_call_and_return_conditional_losses_1100
A__inference_model_5_layer_call_and_return_conditional_losses_1089
A__inference_model_5_layer_call_and_return_conditional_losses_1030
A__inference_model_5_layer_call_and_return_conditional_losses_1019�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
__inference__wrapped_model_962�
���
FullArgSpec
args� 
varargsjargs
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *+�(
&�#
onnx__Gemm_0���������
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkwjkwargs
defaults� 

kwonlyargs�

jtraining%
kwonlydefaults�

trainingp 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkwjkwargs
defaults� 

kwonlyargs�

jtraining%
kwonlydefaults�

trainingp 
annotations� *
 
�2�
.__inference_onnx__Sigmoid_3_layer_call_fn_1131�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
I__inference_onnx__Sigmoid_3_layer_call_and_return_conditional_losses_1124�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
 __inference_4_layer_call_fn_1141�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
;__inference_4_layer_call_and_return_conditional_losses_1136�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
6B4
"__inference_signature_wrapper_1075onnx__Gemm_0�
&__inference_model_5_layer_call_fn_1066]=�:
3�0
&�#
onnx__Gemm_0���������
p 

 
� "�����������
&__inference_model_5_layer_call_fn_1114W7�4
-�*
 �
inputs���������
p 

 
� "�����������
A__inference_model_5_layer_call_and_return_conditional_losses_1030j=�:
3�0
&�#
onnx__Gemm_0���������
p 

 
� "%�"
�
0���������
� �
;__inference_4_layer_call_and_return_conditional_losses_1136X/�,
%�"
 �
inputs���������
� "%�"
�
0���������
� �
__inference__wrapped_model_962b5�2
+�(
&�#
onnx__Gemm_0���������
� "%�"
 
4�
4����������
A__inference_model_5_layer_call_and_return_conditional_losses_1089d7�4
-�*
 �
inputs���������
p

 
� "%�"
�
0���������
� �
.__inference_onnx__Sigmoid_3_layer_call_fn_1131O/�,
%�"
 �
inputs���������
� "�����������
A__inference_model_5_layer_call_and_return_conditional_losses_1100d7�4
-�*
 �
inputs���������
p 

 
� "%�"
�
0���������
� �
"__inference_signature_wrapper_1075rE�B
� 
;�8
6
onnx__Gemm_0&�#
onnx__Gemm_0���������"%�"
 
4�
4����������
I__inference_onnx__Sigmoid_3_layer_call_and_return_conditional_losses_1124\/�,
%�"
 �
inputs���������
� "%�"
�
0���������
� �
A__inference_model_5_layer_call_and_return_conditional_losses_1019j=�:
3�0
&�#
onnx__Gemm_0���������
p

 
� "%�"
�
0���������
� �
&__inference_model_5_layer_call_fn_1107W7�4
-�*
 �
inputs���������
p

 
� "����������o
 __inference_4_layer_call_fn_1141K/�,
%�"
 �
inputs���������
� "�����������
&__inference_model_5_layer_call_fn_1047]=�:
3�0
&�#
onnx__Gemm_0���������
p

 
� "����������