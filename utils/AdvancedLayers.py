import numpy as np
import tensorflow as tf

from tensorflow.python.keras import backend as K
from tensorflow.python.keras.layers import Layer, Lambda
from tensorflow.python.keras.layers import InputSpec
from tensorflow.python.ops import nn_ops
from tensorflow.python.keras import initializers, regularizers, constraints, activations
from tensorflow.python.keras.utils import conv_utils

class Covn2DBaseLayer(Layer):
    """Basic Conv2D class from which other layers inherit.
    """
    def __init__(self,
                 kernel_size,
                 strides=(1, 1),
                 padding='valid',
                 #data_format=None,
                 dilation_rate=(1, 1),
                 activation=None,
                 use_bias=False,
                 kernel_initializer='glorot_uniform',
                 kernel_regularizer=None,
                 kernel_constraint=None,
                 bias_initializer='zeros',
                 bias_regularizer=None,
                 bias_constraint=None,
                 activity_regularizer=None,
                 **kwargs):

        super(Covn2DBaseLayer, self).__init__(
            activity_regularizer=regularizers.get(activity_regularizer), **kwargs)

        self.rank = rank = 2
        self.kernel_size = conv_utils.normalize_tuple(kernel_size, rank, 'kernel_size')
        self.strides = conv_utils.normalize_tuple(strides, rank, 'strides')
        self.padding = conv_utils.normalize_padding(padding)
        self.dilation_rate = conv_utils.normalize_tuple(dilation_rate, rank, 'dilation_rate')
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_initializer = initializers.get(bias_initializer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.bias_constraint = constraints.get(bias_constraint)

    def get_config(self):
        config = super(Covn2DBaseLayer, self).get_config()
        config.update({
            'kernel_size': self.kernel_size,
            'strides': self.strides,
            'padding': self.padding,
            'dilation_rate': self.dilation_rate,
            'activation': activations.serialize(self.activation),
            'use_bias': self.use_bias,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'kernel_constraint': constraints.serialize(self.kernel_constraint),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
            'bias_constraint': constraints.serialize(self.bias_constraint),
            'activity_regularizer': regularizers.serialize(self.activity_regularizer),
        })
        return config

class GroupConv2D(Covn2DBaseLayer):
    """2D Group Convolution layer that shares weights over symmetries.
    
    Group Convolution provides discrete rotation equivariance. It reduces the number 
    of parameters and typically lead to better results.
    
    The following two finite groups are supported:
        Cyclic Group C4 (p4, 4 rotational symmetries)
        Dihedral Group D4 (p4m, 4 rotational and 4 reflection symmetries)
    
    # Arguments
        They are the same as for the normal Conv2D layer.
        filters: int, The effective number of filters is this value multiplied by the
            number of transformations in the group (4 for C4 and 8 for D4)
        kernel_size: int, Only odd values are supported
        group: 'C4' or 'D4', Stay with one group when stacking layers
        
    # Input shape
        featurs: 4D tensor with shape (batch_size, rows, cols, in_channels)
            or 5D tensor with shape (batch_size, rows, cols, num_transformations, in_channels)
    
    # Output shape
        featurs: 5D tensor with shape (batch_size, rows, cols, num_transformations, out_channels)
    
    # Notes
        - BatchNormalization works as expected and shares the statistict over symmetries.
        - Spatial Pooling can be done via AvgPool3D.
        - Pooling along the group dimension can be done via MaxPool3D.
        - Concatenation along the group dimension can be done via Reshape.
        - To get a model with the inference time of a normal CNN, you can load the
          expanded kernel into a normal Conv2D layer. The kernel expansion is
          done in the 'call' method and the expanded kernel is stored in the
          'transformed_kernel' attribute.
    
    # Example
        x = Input((16,16,3))
        x = GroupConv2D(12, 3, group='D4', padding='same', activation='relu')(x)
        x = BatchNormalization()(x)
        x = GroupConv2D(12, 3, group='D4', padding='same', activation='relu')(x)
        x = AvgPool3D(pool_size=(2,2,1), strides=(2,2,1), padding='same')(x)
        x = GroupConv2D(12, 3, group='D4', padding='same', activation='relu')(x)
        x = MaxPool3D(pool_size=(1,1,x.shape[-2]))(x)
        s = x.shape
        x = Reshape((s[1],s[2],s[3]*s[4]))(x)
        
    # References
        [Group Equivariant Convolutional Networks](https://arxiv.org/abs/1602.07576)
        [Rotation Equivariant CNNs for Digital Pathology](https://arxiv.org/abs/1806.03962)
        
        https://github.com/tscohen/GrouPy
        https://github.com/basveeling/keras-gcnn
    """
    
    def __init__(self, filters, kernel_size, group='D4', **kwargs):
        super(GroupConv2D, self).__init__(kernel_size, **kwargs)
        
        if not self.kernel_size[0] == self.kernel_size[1]:
            raise ValueError('Requires square kernel')
        if self.kernel_size[0] % 2 != 1:
            raise ValueError('Requires odd kernel size')
        
        group = group.upper()
        if group == 'C4':
            self.num_transformations = 4
        elif group == 'D4':
            self.num_transformations = 8
        else:
            raise ValueError('Unknown group')
        
        self.filters = filters
        self.group = group
        
        self.input_spec = InputSpec(min_ndim=4, max_ndim=5)
    
    def compute_output_shape(self, input_shape):
        space = input_shape[1:3]
        new_space = []
        for i in range(len(space)):
            new_dim = conv_utils.conv_output_length(
                space[i],
                self.kernel_size[i],
                padding=self.padding,
                stride=self.strides[i],
                dilation=self.dilation_rate[i])
            new_space.append(new_dim)
        return (input_shape[0], *new_space, self.num_transformations, self.filters)
    
    def build(self, input_shape):
        
        if len(input_shape) == 4:
            self.first = True
            num_in_channels = input_shape[-1]
        else:
            self.first = False
            num_in_channels = input_shape[-2] * input_shape[-1]
        
        self.kernel = self.add_weight(name='kernel',
                        shape=(*self.kernel_size, num_in_channels, self.filters),
                        initializer=self.kernel_initializer,
                        regularizer=self.kernel_regularizer,
                        constraint=self.kernel_constraint,
                        trainable=True,
                        dtype=self.dtype)
        
        if self.use_bias:
            self.bias = self.add_weight(name='bias',
                            shape=(self.filters,),
                            initializer=self.bias_initializer,
                            regularizer=self.bias_regularizer,
                            constraint=self.bias_constraint,
                            trainable=True,
                            dtype=self.dtype)
        else:
            self.bias = None
        
        self.built = True
    
    def call(self, features):
        ni = features.shape[-1]
        no = self.filters
        
        if self.group == 'C4':
            nt = 4
        elif self.group == 'D4':
            nt = 8
            
        nti = 1 if self.first else nt
        nto = nt
        
        k = self.kernel_size[0]
        t = np.reshape(np.arange(nti*k*k), (nti,k,k))
        trafos = [np.rot90(t,k,axes=(1, 2)) for k in range(4)]
        if nt == 8:
            trafos = trafos + [np.flip(t,1) for t in trafos]
        self.trafos = trafos = np.array(trafos)
        
        # index magic happens here
        if nti == 1:
            indices = trafos
        elif nti == 4:
            indices = [[trafos[l, (m-l)%4 ,:,:] for m in range(4)] for l in range(4)]
        elif nti == 8:
            indices = [[trafos[l, (m-l)%4 if ((m < 4) == (l < 4)) else (m+l)%4+4 ,:,:] for m in range(8)] for l in range(8)]
        self.indices = indices = np.reshape(indices, (nto,nti,k,k))
        
        # transform the kernel
        kernel = self.kernel
        kernel = tf.reshape(kernel, (nti*k*k, ni, no))
        kernel = tf.gather(kernel, indices, axis=0)
        kernel = tf.reshape(kernel, (nto, nti, k,k, ni, no))
        kernel = tf.transpose(kernel, (2,3,1,4,0,5))
        kernel = tf.reshape(kernel, (k,k, nti*ni, nto*no))
        self.transformed_kernel = kernel
        
        if self.first:
            x = features
        else:
            s = features.shape
            x = tf.reshape(features, (-1,s[1],s[2],s[3]*s[4]))
        
        x = K.conv2d(x, kernel, strides=self.strides, padding=self.padding, dilation_rate=self.dilation_rate)
        s = x.shape
        x = tf.reshape(x, (-1,s[1],s[2],nto,no))
        features = x
        
        if self.use_bias:
            features = tf.add(features, self.bias)
        
        if self.activation is not None:
            features = self.activation(features)
        
        return features
    
    def get_config(self):
        config = super(GroupConv2D, self).get_config()
        config.update({
            'filters': self.filters,
            'group': self.group,
        })
        return config


def rotation_matrix(axis, theta):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.
    """
    axis = np.asarray(axis)
    axis = axis / math.sqrt(np.dot(axis, axis))
    a = math.cos(theta / 2.0)
    b, c, d = -axis * math.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array(
        [
            [aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
            [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
            [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc],
        ]
    )


def equi_coord(pano_W, pano_H, k_W, k_H, u, v):
    fov_w = k_W * np.deg2rad(360.0 / float(pano_W))
    focal = (float(k_W) / 2) / np.tan(fov_w / 2)
    c_x = 0
    c_y = 0

    u_r, v_r = u, v
    u_r, v_r = u_r - float(pano_W) / 2.0, v_r - float(pano_H) / 2.0
    phi, theta = u_r / (pano_W) * (np.pi) * 2, -v_r / (pano_H) * (np.pi)

    ROT = rotation_matrix((0, 1, 0), phi)
    ROT = np.matmul(ROT, rotation_matrix((1, 0, 0), theta))  # np.eye(3)

    h_range = np.array(range(k_H))
    w_range = np.array(range(k_W))
    w_ones = np.ones(k_W)
    h_ones = np.ones(k_H)
    h_grid = (
            np.matmul(np.expand_dims(h_range, -1), np.expand_dims(w_ones, 0))
            + 0.5
            - float(k_H) / 2
    )
    w_grid = (
            np.matmul(np.expand_dims(h_ones, -1), np.expand_dims(w_range, 0))
            + 0.5
            - float(k_W) / 2
    )

    K = np.array([[focal, 0, c_x], [0, focal, c_y], [0.0, 0.0, 1.0]])
    inv_K = np.linalg.inv(K)
    rays = np.stack([w_grid, h_grid, np.ones(h_grid.shape)], 0)
    rays = np.matmul(inv_K, rays.reshape(3, k_H * k_W))
    rays /= np.linalg.norm(rays, axis=0, keepdims=True)
    rays = np.matmul(ROT, rays)
    rays = rays.reshape((3, k_H, k_W))

    phi = np.arctan2(rays[0, ...], rays[2, ...])
    theta = np.arcsin(np.clip(rays[1, ...], -1, 1))
    x = (pano_W) / (2.0 * np.pi) * phi + float(pano_W) / 2.0
    y = (pano_H) / (np.pi) * theta + float(pano_H) / 2.0

    roi_y = h_grid + v_r + float(pano_H) / 2.0
    roi_x = w_grid + u_r + float(pano_W) / 2.0

    new_roi_y = y
    new_roi_x = x

    offsets_x = new_roi_x - roi_x
    offsets_y = new_roi_y - roi_y

    return offsets_x, offsets_y


def equi_coord_fixed_resoltuion(pano_W, pano_H, k_W, k_H, u, v, pano_Hf=-1, pano_Wf=-1):
    pano_Hf = pano_H if pano_Hf <= 0 else pano_H / pano_Hf
    pano_Wf = pano_W if pano_Wf <= 0 else pano_W / pano_Wf
    fov_w = k_W * np.deg2rad(360.0 / float(pano_Wf))
    focal = (float(k_W) / 2) / np.tan(fov_w / 2)
    c_x = 0
    c_y = 0

    u_r, v_r = u, v
    u_r, v_r = u_r - float(pano_W) / 2.0, v_r - float(pano_H) / 2.0
    phi, theta = u_r / (pano_W) * (np.pi) * 2, -v_r / (pano_H) * (np.pi)

    ROT = rotation_matrix((0, 1, 0), phi)
    ROT = np.matmul(ROT, rotation_matrix((1, 0, 0), theta))  # np.eye(3)

    h_range = np.array(range(k_H))
    w_range = np.array(range(k_W))
    w_ones = np.ones(k_W)
    h_ones = np.ones(k_H)
    h_grid = (
            np.matmul(np.expand_dims(h_range, -1), np.expand_dims(w_ones, 0))
            + 0.5
            - float(k_H) / 2
    )
    w_grid = (
            np.matmul(np.expand_dims(h_ones, -1), np.expand_dims(w_range, 0))
            + 0.5
            - float(k_W) / 2
    )

    K = np.array([[focal, 0, c_x], [0, focal, c_y], [0.0, 0.0, 1.0]])
    inv_K = np.linalg.inv(K)
    rays = np.stack([w_grid, h_grid, np.ones(h_grid.shape)], 0)
    rays = np.matmul(inv_K, rays.reshape(3, k_H * k_W))
    rays /= np.linalg.norm(rays, axis=0, keepdims=True)
    rays = np.matmul(ROT, rays)
    rays = rays.reshape((3, k_H, k_W))

    phi = np.arctan2(rays[0, ...], rays[2, ...])
    theta = np.arcsin(np.clip(rays[1, ...], -1, 1))
    x = (pano_W) / (2.0 * np.pi) * phi + float(pano_W) / 2.0
    y = (pano_H) / (np.pi) * theta + float(pano_H) / 2.0

    roi_y = h_grid + v_r + float(pano_H) / 2.0
    roi_x = w_grid + u_r + float(pano_W) / 2.0

    new_roi_y = y
    new_roi_x = x

    offsets_x = new_roi_x - roi_x
    offsets_y = new_roi_y - roi_y

    return offsets_x, offsets_y


def distortion_aware_map(pano_W, pano_H, k_W, k_H, s_width=10, s_height=10, bs=16):
    offset = np.zeros(shape=[pano_H, pano_W, k_H * k_W * 2])

    for v in range(0, pano_H, s_height):
        u = 0
        offsets_x, offsets_y = equi_coord_fixed_resoltuion(
            pano_W, pano_H, k_W, k_H, u, v, 1, 1
        )

        # lower edge
        for i in range(1, k_W - 1):
            offsets_y[-1][i] = offsets_y[-1][0]

        # left edge
        k = (offsets_x[0][0] - offsets_x[-1][0]) / (offsets_y[0][0] - offsets_y[-1][0])
        c = offsets_x[0][0] - k * offsets_y[0][0]
        for i in range(1, k_H - 1):
            offsets_x[i][0] = k * offsets_y[i][0] + c

        # right edge
        k = (offsets_x[0][-1] - offsets_x[-1][-1]) / (offsets_y[0][-1] - offsets_y[-1][-1])
        c = offsets_x[0][-1] - k * offsets_y[0][-1]
        for i in range(1, k_H - 1):
            offsets_x[i][-1] = k * offsets_y[i][-1] + c

        offsets = np.concatenate(
            (np.expand_dims(offsets_y, -1), np.expand_dims(offsets_x, -1)), axis=-1
        )
        total_offsets = offsets.flatten().astype("float32")
        offset[v, u, :] = total_offsets

        for v_ in range(s_height):
            for u_ in range(pano_W):
                try:
                    offset[v + v_, u + u_, :] = total_offsets
                except:
                    pass

    offset = tf.constant(offset)
    offset = tf.expand_dims(offset, 0)
    offset = tf.tile(offset, multiples=[bs, 1, 1, 1])
    offset = tf.cast(offset, tf.float32)

    return offset


class CHConv(tf.keras.layers.Layer):
    @typechecked
    def __init__(
            self,
            filters: int,
            kernel_size: tuple = (3, 3),
            num_groups: int = 1,
            deformable_groups: int = 1,
            strides: tuple = (1, 1),
            im2col: int = 1,
            s_strides: int = 10,
            use_bias: bool = False,
            padding: str = "valid",
            data_format: str = "channels_last",
            dilations: tuple = (1, 1),
            use_relu: bool = False,
            kernel_initializer: types.Initializer = None,
            kernel_regularizer: types.Regularizer = None,
            kernel_constraint: types.Constraint = None,
            **kwargs
    ):
        super(CHConv, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.num_groups = num_groups
        self.deformable_groups = deformable_groups
        self.strides = strides
        self.im2col = im2col
        self.s_strides = s_strides
        self.use_bias = use_bias
        self.padding = padding
        self.data_format = data_format
        self.dilations = dilations
        self.use_relu = use_relu
        self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self.kernel_regularizer = tf.keras.regularizers.get(kernel_regularizer)
        self.kernel_constraint = tf.keras.constraints.get(kernel_constraint)
        if self.padding == "valid":
            self.tf_pad = "VALID"
        else:
            self.tf_pad = "SAME"

    def build(self, input_shape):
        if self.data_format == "channels_last":
            channel = int(input_shape[-1])
        else:
            channel = int(input_shape[1])
        self.kernel = self.add_weight(
            shape=[self.filters, channel, self.kernel_size[0], self.kernel_size[1]],
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            trainable=True,
            name="w"
        )
        self.scale = self.add_weight(
            shape=[input_shape[1], input_shape[2], self.kernel_size[0] * self.kernel_size[1] * 2],
            initializer=tf.keras.initializers.Constant(value=1.),
            regularizer=self.kernel_regularizer,
            constraint=tf.keras.constraints.MinMaxNorm(min_value=0.5, max_value=1.5, rate=1.0, axis=0),
            trainable=True,
            name="s"
        )

        if self.use_bias:
            self.bias = self.add_weight(
                shape=[1, self.filters, 1, 1],
                initializer=self.kernel_initializer,
                regularizer=self.kernel_regularizer,
                constraint=self.kernel_constraint,
                trainable=True,
            )
        self.built = True

    def compute_output_shape(self, input_shape):
        input_shape = tf.TensorShape(input_shape).as_list()
        if self.data_format == "channels_last":
            space = input_shape[1:-1]
            new_space = []
            for i in range(len(space)):
                new_dim = conv_utils.conv_output_length(
                    space[i],
                    self.kernel_size[i],
                    padding=self.padding,
                    stride=self.strides[i],
                    dilation=self.dilation_rate[i],
                )
                new_space.append(new_dim)
            return tf.TensorShape([input_shape[0]] + new_space + [self.filters])
        else:
            space = input_shape[2:]
            new_space = []
            for i in range(len(space)):
                new_dim = conv_utils.conv_output_length(
                    space[i],
                    self.kernel_size[i],
                    padding=self.padding,
                    stride=self.strides[i],
                    dilation=self.dilation_rate[i],
                )
                new_space.append(new_dim)
            return tf.TensorShape([input_shape[0], self.filters] + new_space)

    def call(self, inputs, **kwargs):
        if self.data_format == "channels_first":
            data = tf.transpose(inputs, [0, 2, 3, 1])
        else:
            data = inputs
        n, h, w, c_i = tuple(data.get_shape().as_list())
        data_shape = tf.shape(data)
        """
        The original implement in paper here bs is set as self.batch_size, here wo use data_shape[0],
        because self.batch_size if constant value and can't changed, but actually image batch_size can
        change in train and test period, so we use tf.shape to get actual dynamic batch_size.
        """

        offset = tf.stop_gradient(
            distortion_aware_map(
                w,
                h,
                self.kernel_size[0],
                self.kernel_size[1],
                s_width=self.s_strides,
                s_height=self.s_strides,
                bs=data_shape[0],
            )
        )
        mask = tf.stop_gradient(
            tf.ones(
                shape=[
                    data_shape[0],
                    data_shape[1],
                    data_shape[2],
                    self.kernel_size[0] * self.kernel_size[1],
                ]
            )
        )

        # offset = tf.keras.layers.Conv2D(filters=self.kernel_size[0] * self.kernel_size[1] * 2, kernel_size=(3, 3), padding='same',use_bias=False)(offset)
        offset = tf.multiply(self.scale, offset)
        data = tf.transpose(data, [0, 3, 1, 2])
        offset = tf.transpose(offset, [0, 3, 1, 2])

        mask = tf.transpose(mask, [0, 3, 1, 2])
        res = _deformable_conv2d(
            data,
            self.kernel,
            offset,
            mask,
            [1, 1, self.strides[0], self.strides[1]],
            num_groups=self.num_groups,
            deformable_groups=self.deformable_groups,
            padding=self.tf_pad,
            data_format="NCHW",
        )
        # print(res.shape)
        if self.use_bias:
            res = tf.add(res, self.bias)
        if self.use_relu:
            res = tf.nn.relu(res)
        if self.data_format == "channels_last":
            return tf.transpose(res, [0, 2, 3, 1])
        else:
            return res

    def get_config(self):
        config = {
            "filters": self.filters,
            "kernel_size": self.kernel_size,
            "num_groups": self.num_groups,
            "deformable_groups": self.deformable_groups,
            "strides": self.strides,
            "im2col": self.im2col,
            "use_bias": self.use_bias,
            "padding": self.padding,
            "data_format": self.data_format,
            "dilations": self.dilations,
            "use_relu": self.use_relu,
            "kernel_initializer": tf.keras.initializers.serialize(
                self.kernel_initializer
            ),
            "kernel_regularizer": tf.keras.regularizers.serialize(
                self.kernel_regularizer
            ),
            "kernel_constraint": tf.keras.constraints.serialize(self.kernel_constraint),
            "tf_pad": self.tf_pad,
        }
        base_config = super().get_config()
        return {**base_config, **config}