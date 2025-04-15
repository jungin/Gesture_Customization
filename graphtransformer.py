import os
import shutil

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, BatchNormalization, Input, Dropout
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint

from datagenerator import CustomDataGen
from customlayer import GetAdjacencyMatrixLayer, GraphTransformerLayer, ReshapeLayer1, ReshapeLayer2

from variables import preprocessed_train_dir, preprocessed_val_dir,\
                      train_csv, val_csv, \
                      JESTER_CLASSES, \
                      preprocessed_train_subset_dir

tf.config.optimizer.set_experimental_options({
    'layout_optimizer': False,
    'constant_folding': True,
    'shape_optimization': True
})

def create_model():
    # Model Architecture
    node_input = Input(shape=(37, 42, 3), batch_size=None, name='NodeInput')
    adjacency_input = GetAdjacencyMatrixLayer(name='AdjacencyMatrix')(node_input)

    attention_outputs = []
    x = node_input

    # GraphTransformer layers
    for i in range(3):
        x, attention_weights = GraphTransformerLayer(
            embed_dim=64, 
            num_heads=8, 
            name=f'GraphTransformerLayer_{i}'
        )(x, adjacency_input)
        attention_outputs.append(attention_weights)
        x = BatchNormalization(name=f'BatchNormalization_{i}')(x)
        x = tf.keras.layers.ReLU(name=f'ReLU_{i}')(x)
        x = tf.keras.layers.SpatialDropout2D(0.1, name=f'GraphTransformerDropout_{i}')(x)

    # Reshape layers
    x = ReshapeLayer1(name='Reshape1')(x)
    x = Dropout(0.2, name='ReshapeDropout1')(x)  
    x = ReshapeLayer2(name='Reshape2')(x)
    x = Dropout(0.2, name='ReshapeDropout2')(x)  

    # Dense layers
    x = Dense(1024, activation="relu", name='Dense1')(x)
    x = BatchNormalization(name='BatchNorm1')(x)
    x = Dropout(0.5, name='Dropout1')(x)

    x = Dense(512, activation="relu", name='Dense2')(x)
    x = BatchNormalization(name='BatchNorm2')(x)
    x = Dropout(0.4, name='Dropout2')(x)

    x = Dense(256, activation="relu", name='Dense3')(x)
    x = BatchNormalization(name='BatchNorm3')(x)
    x = Dropout(0.3, name='Dropout3')(x)

    output = Dense(27, activation="softmax", name='Output')(x)

    # Model definition
    outputs = {
        'Output': output,
        'GraphTransformerLayer_0': attention_outputs[0],
        'GraphTransformerLayer_1': attention_outputs[1],
        'GraphTransformerLayer_2': attention_outputs[2]
    }
    model = Model(inputs=node_input, outputs=outputs)
    
    # Model compilation
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss={
            'Output': 'categorical_crossentropy',
            'GraphTransformerLayer_0': None,
            'GraphTransformerLayer_1': None,
            'GraphTransformerLayer_2': None
        },
        loss_weights={
            'Output': 1.0,
            'GraphTransformerLayer_0': 0.0,
            'GraphTransformerLayer_1': 0.0,
            'GraphTransformerLayer_2': 0.0
        },
        metrics={
            'Output': [tf.keras.metrics.CategoricalAccuracy(name='accuracy')],
            'GraphTransformerLayer_0': None,
            'GraphTransformerLayer_1': None,
            'GraphTransformerLayer_2': None
        },
        run_eagerly=True
    )
    
    return model

def main():
    batch_size = 32
    file_path = 'best_model_2'

    # 기존 저장 디렉토리가 있다면 삭제
    if os.path.exists(file_path):
        shutil.rmtree(file_path)

    # Data loader setup
    # 데이터 형태 설정 (예시)
    num_frames = 30      # 비디오 프레임 수
    num_joints = 17      # 관절 포인트 수
    coordinate_dim = 3   # 좌표 차원 (x, y, z)

    # 데이터 생성기 인스턴스 생성
    traindata = CustomDataGen(
        csv=train_csv,
        dir=preprocessed_train_dir,
        batch_size=batch_size,
        shuffle=True,
        num_frames=num_frames,
        num_joints=num_joints,
        coordinate_dim=coordinate_dim
    )

    valdata = CustomDataGen(
        csv=val_csv,
        dir=preprocessed_val_dir,
        batch_size=batch_size,
        shuffle=True,
        num_frames=num_frames,
        num_joints=num_joints,
        coordinate_dim=coordinate_dim
    )

    # traindata = CustomDataGen(train_csv, preprocessed_train_dir, batch_size=batch_size, shuffle=True)
    # valdata = CustomDataGen(val_csv, preprocessed_val_dir, batch_size=batch_size, shuffle=True)

    # Create and compile model
    model = create_model()
    model.build(input_shape=(None, 37, 42, 3))
    model.summary()

    # Callbacks setup
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        verbose=1
    )

    checkpoint = ModelCheckpoint(
        filepath=file_path,
        monitor='val_loss',
        save_best_only=True,
        save_format='tf',
        verbose=1,
        mode='min'
    )

    earlystopping = EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True,
        verbose=1
    )

    callbacks = [reduce_lr, checkpoint, earlystopping]

    # Training
    print("Starting model training...")
    hist = model.fit(
        traindata,
        validation_data=valdata,
        epochs=100,
        callbacks=callbacks,
        verbose=1
    )

    # plot loss and accuracy
    import matplotlib.pyplot as plt
    import seaborn as sns

    # 스타일 설정
    sns.set_style("whitegrid")
    sns.set_palette("husl")

    # 그래프 그리기
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # 손실 그래프
    sns.lineplot(data=hist.history['loss'], label='Training Loss', ax=ax1)
    sns.lineplot(data=hist.history['val_loss'], label='Validation Loss', ax=ax1)
    ax1.set_title('Model Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')

    # 정확도 그래프
    sns.lineplot(data=hist.history['Output_accuracy'], label='Training Accuracy', ax=ax2)
    sns.lineplot(data=hist.history['val_Output_accuracy'], label='Validation Accuracy', ax=ax2)
    ax2.set_title('Model Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')

    plt.tight_layout()
    plt.savefig('graph_transformer_result.png')


    print("\nTraining completed. Loading best model...")
    # 저장된 최상의 모델 로드
    loaded_model = tf.keras.models.load_model(file_path)

    # 모델 평가
    print("\nEvaluating loaded model on validation data...")
    evaluation = loaded_model.evaluate(valdata, verbose=1)
    print(f"\nValidation loss: {evaluation[0]:.4f}")
    print(f"Validation accuracy: {evaluation[1]:.4f}")

if __name__ == "__main__":
    main()