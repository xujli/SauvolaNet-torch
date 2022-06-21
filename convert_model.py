import tensorflow as tf
from SauvolaDocBin.modelUtils import *
from pytorch_edition.modelUtils import *
from SauvolaDocBin.layerUtils import *
from SauvolaDocBin.layerUtils import SauvolaLayerObjects

# metrics=['TextAcc', 'Acc', 'F1', 'PSNR']
#
model_filepath = './pretrained_models/Sauvola_v3_att_w7.15.23.31.39.47.55.63_k1_R1_a1_inorm_S256_R0.02_B8_lr0.001_E30-Acc0.9895-Tacc0.9282-F0.9448-PSNR20.59.h5'
model = tf.keras.models.load_model(model_filepath, compile=True,
                                   custom_objects=SauvolaLayerObjects)
#
# model.trainable = False
#
# print(model.trainable)
# model.summary()
# net = Multiscale_sauvola()
def keras_to_pyt(km, pm):
    weight_dict = dict()
    for layer in km.layers:
        if len(layer.get_weights()) > 0:
            if type(layer) is tf.keras.layers.Conv2D:
                weight_dict[layer.get_config()['name'] + '.weight'] = np.transpose(layer.get_weights()[0], (3, 2, 0, 1))
                weight_dict[layer.get_config()['name'] + '.bias'] = layer.get_weights()[1]
            elif isinstance(layer, SauvolaMultiWindow):
                weight_dict[layer.get_config()['name'] + '.k'] = layer.get_weights()[0]
                weight_dict[layer.get_config()['name'] + '.R'] = layer.get_weights()[1]
            elif isinstance(layer, DifferenceThresh):
                weight_dict[layer.get_config()['name'] + '.alpha'] = layer.get_weights()[0]
            else:
                print('No match')

    pyt_state_dict = pm.state_dict()
    for key1, key2 in zip(pyt_state_dict.keys(), weight_dict.keys()):
        if ('conv' in key1) and ('weight' in key1):
            tensor = torch.from_numpy(weight_dict[key2])
            pyt_state_dict[key1] = tensor
        else:
            pyt_state_dict[key1] = torch.from_numpy(weight_dict[key2])

    return pyt_state_dict

# net.load_state_dict(keras_to_pyt(model, net))
# for (name, param), weight in zip(net.named_parameters(), model.get_weights()):
#     if len(weight.shape) == 4:
#         print(name, np.mean(param.detach().numpy().flatten()), np.mean(np.array(weight).flatten()),
#             np.mean(np.abs(param.detach().numpy() - np.transpose(np.array(weight), (3, 2, 0, 1)))))
#
#
# net.eval()
# dense1_layer_model = tf.keras.Model(inputs=model.input, outputs=[model.get_layer('difference_thresh').output,
#                                                                  model.get_layer('conv_att').output])
#
# data = np.random.random((1, 1, 256, 256)).astype(np.float32)
# output1 = dense1_layer_model.predict(np.transpose(data, (0, 2, 3, 1)))
# output3 = net(torch.from_numpy(data))
# for item1, item2 in zip(output3, output1):
#     if item1.detach().numpy().shape != item2.shape:
#         print(np.nanmean(np.abs(np.transpose(item1.detach().numpy(), (0, 2, 3, 1)) - np.array(item2))), np.nanmean(np.array(item2)))
#     else:
#         print(np.nanmean(np.abs(item1.detach().numpy() - np.array(item2))), np.nanmean(np.array(item2)))
