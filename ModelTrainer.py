from DataGenerator import SimpleDataGeneratorGetter
from VGG16AutoEncoderGenerator import VGG16AutoEncoderGenerator
from keras.callbacks import ModelCheckpoint
import os


def save_model_as_json(model, target_file_path):
    model_in_json = model.to_json()
    with open(target_file_path, "w") as file:
        file.write(model_in_json)


if __name__ == '__main__':
    input_shape = (224, 224, 3)
    model_and_weight_storing_dir = "model_and_weight"

    vgg16_auto_encoder_generator = VGG16AutoEncoderGenerator(
        input_shape=input_shape,
        weight_file=os.path.join(model_and_weight_storing_dir, 'vgg16_weights_notop.h5'),
        number_of_layers_to_freeze_from_input_layer=10
    )

    vgg16_auto_encoder_generator.auto_encoder.compile(
        loss='binary_crossentropy',
        optimizer='adadelta',
        metrics=['accuracy']
    )

    base_data_source_dir = 'data_source'

    vgg16_auto_encoder_generator.auto_encoder.fit_generator(
        generator=SimpleDataGeneratorGetter().get_generator(
            data_source_dir=os.path.join(base_data_source_dir, "training"),
            batch_size=64
        ).infinitely_generate_batch_of_data_pair_tuple(),
        steps_per_epoch=300, epochs=150, verbose=2,
        callbacks=[
            ModelCheckpoint(
                filepath=os.path.join(model_and_weight_storing_dir, "auto_encoder_model_weight_checkpoint.h5"),
                save_weights_only=True, verbose=1
            )
        ]
    )

    print(vgg16_auto_encoder_generator.auto_encoder.metrics_names)
    scores = vgg16_auto_encoder_generator.auto_encoder.evaluate_generator(
        generator=SimpleDataGeneratorGetter().get_generator(
            data_source_dir=os.path.join(base_data_source_dir, "testing"),
            batch_size=100
        ).infinitely_generate_batch_of_data_pair_tuple(),
        steps=100
    )
    print(scores)

    print("save encoder as 'encoder_model.json' and 'encoder_weight.h5'")
    save_model_as_json(
        vgg16_auto_encoder_generator.encoder, os.path.join(model_and_weight_storing_dir, "encoder_model.json")
    )
    vgg16_auto_encoder_generator.encoder.save_weights(
        os.path.join(model_and_weight_storing_dir, 'encoder_weight.h5')
    )

    print("save auto encoder as 'auto_encoder_model.json' and 'auto_encoder_weight.h5")
    save_model_as_json(
        vgg16_auto_encoder_generator.auto_encoder, os.path.join(model_and_weight_storing_dir, "auto_encoder_model.json")
    )
    vgg16_auto_encoder_generator.auto_encoder.save_weights(
        os.path.join(model_and_weight_storing_dir, 'auto_encoder_weight.h5')
    )
