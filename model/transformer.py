# 재사용을 위해

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import re


def preprocess_sentence(sentence):
    sentence = sentence.lower()  # 소문자화

    sentence = re.sub(r"([?.!,])", r" \1 ", sentence)  # 기호처리
    sentence = re.sub(r'[" "]+', " ", sentence)  # 연속 공백 처리
    sentence = re.sub(r"[^가-힣a-zA-Z?.!,]+", " ", sentence)  # 기타 문자 제거

    sentence = sentence.strip()  # 양쪽 공백 제거

    return sentence


# 포지셔널 인코딩
def positional_encoding(pos, d_model):
    def cal_angle(position, i):
        return position / np.power(10000, int(i) / d_model)

    def get_posi_angle_vec(position):
        return [cal_angle(position, i) for i in range(d_model)]

    sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in range(pos)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])
    return sinusoid_table


# 마스크
def generate_padding_mask(seq):
    # 실수가 아닌 것을 전부 0으로
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
    return seq[:, tf.newaxis, tf.newaxis, :]


def generate_lookahead_mask(size):
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask


def generate_masks(src, tgt):
    enc_mask = generate_padding_mask(src)
    dec_enc_mask = generate_padding_mask(src)

    dec_lookahead_mask = generate_lookahead_mask(tgt.shape[1])
    dec_tgt_padding_mask = generate_padding_mask(tgt)
    dec_mask = tf.maximum(dec_tgt_padding_mask, dec_lookahead_mask)

    return enc_mask, dec_enc_mask, dec_mask


# 멀티헤드 어텐션
class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        self.depth = d_model // self.num_heads

        self.W_q = tf.keras.layers.Dense(d_model)  # Linear Layer
        self.W_k = tf.keras.layers.Dense(d_model)
        self.W_v = tf.keras.layers.Dense(d_model)

        self.linear = tf.keras.layers.Dense(d_model)

    def scaled_dot_product_attention(self, Q, K, V, mask):
        d_k = tf.cast(K.shape[-1], tf.float32)  # 나누기 위하여 타입 변경

        scaled_qk = tf.matmul(Q, K, transpose_b=True)  # q, k의 유사도 측정
        scaled_qk /= tf.sqrt(d_k)  # 스케일 조정

        # 마스크에 해당하는 아주 큰 값을 빼서 softmax를 통과하면 0에 수렴하게 만듦
        if mask is not None:
            scaled_qk += mask * -1e9

        attentions = tf.nn.softmax(scaled_qk, -1)
        out = tf.matmul(attentions, V)  # 어텐션 가중치 반영

        return out, attentions

    def split_heads(self, x):
        """
        Embedding을 Head의 수로 분할하는 함수

        x: [ batch x length x emb ]
        return: [ batch x heads x length x self.depth ]
        """

        split_x = tf.reshape(x, (x.shape[0], -1, self.num_heads, self.depth))
        split_x = tf.transpose(split_x, [0, 2, 1, 3])

        return split_x

    def combine_heads(self, x):
        """
        분할된 Embedding을 하나로 결합하는 함수

        x: [ batch x heads x length x self.depth ]
        return: [ batch x length x emb ]
        """
        # split_heads의 역순 진행
        combined_x = tf.transpose(x, [0, 2, 1, 3])
        combined_x = tf.reshape(
            combined_x, (combined_x.shape[0], -1, self.d_model)
        )

        return combined_x

    def call(self, Q, K, V, mask):
        # 리니어 적용
        WQ = self.W_q(Q)
        WK = self.W_k(K)
        WV = self.W_v(V)

        # 머리 쪼개기
        WQ_split = self.split_heads(WQ)
        WK_split = self.split_heads(WK)
        WV_split = self.split_heads(WV)

        # 어텐션
        out, attention_weights = self.scaled_dot_product_attention(
            WQ_split, WK_split, WV_split, mask
        )

        # 머리 합치기
        out = self.combine_heads(out)
        out = self.linear(out)

        return out, attention_weights


# FFN
class PoswiseFeedForwardNet(tf.keras.layers.Layer):
    def __init__(self, d_model, d_ff):
        super(PoswiseFeedForwardNet, self).__init__()
        self.w_1 = tf.keras.layers.Dense(d_ff, activation="relu")
        self.w_2 = tf.keras.layers.Dense(d_model)

    def call(self, x):
        out = self.w_1(x)
        out = self.w_2(out)

        return out


### 인코더 디코더
# 인코더층
class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, n_heads, d_ff, dropout):
        super(EncoderLayer, self).__init__()

        self.enc_self_attn = MultiHeadAttention(d_model, n_heads)
        self.ffn = PoswiseFeedForwardNet(d_model, d_ff)

        self.norm_1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.norm_2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout = tf.keras.layers.Dropout(dropout)

    def call(self, x, mask):
        # Multi-Head Attention
        residual = x
        out = self.norm_1(x)
        out, enc_attn = self.enc_self_attn(out, out, out, mask)
        out = self.dropout(out)
        out += residual

        # Position-Wise Feed Forward Networn
        residual = out
        out = self.norm_2(out)
        out = self.ffn(out)
        out = self.dropout(out)
        out += residual

        return out, enc_attn


# 디코더층
class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, n_heads, d_ff, dropout):
        super(DecoderLayer, self).__init__()

        self.dec_self_attn = MultiHeadAttention(d_model, n_heads)
        self.enc_dec_attn = MultiHeadAttention(d_model, n_heads)
        self.ffn = PoswiseFeedForwardNet(d_model, d_ff)

        self.norm_1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.norm_2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.norm_3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout = tf.keras.layers.Dropout(dropout)

    def call(self, x, enc_out, mask, casual_mask):

        # Multi-Head Attention
        residual = x
        out = self.norm_1(x)
        out, dec_attn = self.dec_self_attn(out, out, out, mask)
        out = self.dropout(out)
        out += residual

        # encoder-decoder Attention
        residual = out
        out = self.norm_2(x)
        out, enc_dec_attn = self.enc_dec_attn(out, enc_out, enc_out, casual_mask)
        out = self.dropout(out)
        out += residual

        # Position-Wise Feed Forward Network
        residual = out
        out = self.norm_3(out)
        out = self.ffn(out)
        out = self.dropout(out)
        out += residual

        return out, dec_attn, enc_dec_attn


# 전체 인코더
class Encoder(tf.keras.Model):
    def __init__(self, n_layers, d_model, n_heads, d_ff, dropout):
        super(Encoder, self).__init__()
        self.n_layers = n_layers
        self.enc_layers = [
            EncoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)
        ]

    def call(self, x, mask):
        out = x

        enc_attns = list()
        for i in range(self.n_layers):
            out, enc_attn = self.enc_layers[i](out, mask)
            enc_attns.append(enc_attn)

        return out, enc_attns


# 전체 디코더
class Decoder(tf.keras.Model):
    def __init__(self, n_layers, d_model, n_heads, d_ff, dropout):
        super(Decoder, self).__init__()
        self.n_layers = n_layers
        self.dec_layers = [
            DecoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)
        ]

    def call(self, x, enc_out, padding_mask, causality_mask):
        out = x

        dec_attns = list()
        dec_enc_attns = list()
        for i in range(self.n_layers):
            out, dec_attn, dec_enc_attn = self.dec_layers[i](
                out, enc_out, padding_mask, causality_mask
            )

            dec_attns.append(dec_attn)
            dec_enc_attns.append(dec_enc_attn)

        return out, dec_attns, dec_enc_attns


### 트랜스포머
class Transformer(tf.keras.Model):
    def __init__(
        self,
        n_layers,
        d_model,
        n_heads,
        d_ff,
        src_vocab_size,
        tgt_vocab_size,
        pos_len,
        dropout=0.2,
        shared_fc=True,
        shared_emb=False,
    ):
        super(Transformer, self).__init__()
        self.d_model = tf.cast(d_model, tf.float32)

        if shared_emb:
            self.enc_embedding = self.dec_embedding = tf.keras.layers.Embedding(
                src_vocab_size, d_model
            )

        else:
            self.enc_embedding = tf.keras.layers.Embedding(src_vocab_size, d_model)
            self.dec_embedding = tf.keras.layers.Embedding(tgt_vocab_size, d_model)

        self.pos_encoding = positional_encoding(pos_len, d_model)
        self.dropout = tf.keras.layers.Dropout(dropout)

        self.encoder = Encoder(n_layers, d_model, n_heads, d_ff, dropout)
        self.decoder = Decoder(n_layers, d_model, n_heads, d_ff, dropout)

        self.fc = tf.keras.layers.Dense(tgt_vocab_size)

        self.shared_fc = shared_fc
        if self.shared_fc:
            self.fc.set_weights(tf.transpose(self.dec_embedding.weights))

    def embedding(self, emb, x):
        """
        입력된 정수 배열을 Embedding + Pos Encoding
        + Shared일 경우 Scaling 작업 포함

        x: [ batch x length ]
        return: [ batch x length x emb ]
        """

        out = emb(x)

        if self.shared_fc:
            out *= tf.sqrt(self.d_model)

        # 포지셔널 인코딩은 batch차원이 없어서 확장 : [tf.newaxis, ...]
        # 데이터 길이 만큼만 더하기 : [:, :x.shape[1], :]
        out += self.pos_encoding[np.newaxis, ...][:, : x.shape[1], :]
        out = self.dropout(out)

        return out

    def call(self, enc_in, dec_in, enc_mask, causality_mask, dec_mask):
        # 임베딩
        enc_in = self.embedding(self.enc_embedding, enc_in)
        dec_in = self.embedding(self.dec_embedding, dec_in)

        # 인코더
        enc_out, enc_attns = self.encoder(enc_in, enc_mask)
        # 디코더
        dec_out, dec_attns, enc_dec_attns = self.decoder(
            dec_in, enc_out, dec_mask, causality_mask
        )
        # 최종 출력
        logits = self.fc(dec_out)

        return logits, enc_attns, dec_attns, enc_dec_attns


### 훈련
# 학습률, 옵티마이저
class LearningRateScheduler(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super(LearningRateScheduler, self).__init__()
        self.d_model = d_model
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = step**-0.5
        arg2 = step * (self.warmup_steps**-1.5)

        return (self.d_model**-0.5) * tf.math.minimum(arg1, arg2)


# loss
def loss_function(
    real,
    pred,
    loss_object=tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction="none"
    ),
):

    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)

    # Masking 되지 않은 입력의 개수로 Scaling
    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_sum(loss_) / tf.reduce_sum(mask)


# Train Step


@tf.function()
def train_step(src, tgt, model, optimizer):
    tgt_in = tgt[:, :-1]
    gold = tgt[:, 1:]

    enc_mask, dec_enc_mask, dec_mask = generate_masks(src, tgt_in)

    # 계산된 loss에 tf.GradientTape()를 적용해 학습을 진행합니다.
    with tf.GradientTape() as tape:
        predictions, enc_attns, dec_attns, dec_enc_attns = model(
            src, tgt_in, enc_mask, dec_enc_mask, dec_mask
        )
        loss = loss_function(gold, predictions)

    # 최종적으로 optimizer.apply_gradients()가 사용됩니다.
    variables = model.trainable_variables
    gradients = tape.gradient(loss, variables)
    optimizer.apply_gradients(zip(gradients, variables))

    return loss, enc_attns, dec_attns, dec_enc_attns


@tf.function()
def valid_step(src, tgt, model):
    tgt_in = tgt[:, :-1]
    gold = tgt[:, 1:]

    enc_mask, dec_enc_mask, dec_mask = generate_masks(src, tgt_in)

    # GradientTape 없이
    predictions, enc_attns, dec_attns, dec_enc_attns = model(
        src, tgt_in, enc_mask, dec_enc_mask, dec_mask
    )

    # 검증 손실 계산
    loss = loss_function(gold, predictions)

    return loss, enc_attns, dec_attns, dec_enc_attns


def validate(dataset, model):
    total_loss = 0
    num_batches = 0

    for src, tgt in dataset:
        loss, _, _, _ = valid_step(src, tgt, model)
        total_loss += loss
        num_batches += 1

    average_loss = total_loss / num_batches
    return average_loss


def train(
    model,
    dataset,
    EPOCH,
    enc_tokenizer,
    dec_tokenizer,
    type=None,
    show_translate=False,
    example_sentence=None,
    plot_attention=False,
    valid_data=None,
):
    import tqdm

    learning_rate = LearningRateScheduler(int(model.d_model.numpy()))
    optimizer = tf.keras.optimizers.Adam(
        learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9
    )

    for epoch in range(EPOCH):
        total_loss = 0

        dataset_count = tf.data.experimental.cardinality(dataset).numpy()
        t = tqdm.tqdm(dataset, total=dataset_count)

        for batch, (enc_train, dec_train) in enumerate(t):
            enc_train = tf.convert_to_tensor(enc_train)
            dec_train = tf.convert_to_tensor(dec_train)

            loss, enc_attns, dec_attns, dec_enc_attns = train_step(
                enc_train,
                dec_train,
                model,
                optimizer,
            )

            total_loss += loss

            t.set_description_str("Epoch %2d" % (epoch + 1))  # tqdm
            t.set_postfix_str("Loss %.4f" % (total_loss.numpy() / (batch + 1)))  # tqdm

        if valid_data:
            loss_val = validate(valid_data, model)
            print("Valid Loss %.4f" % (loss_val.numpy()))

        if example_sentence:
            if type == "sp":
                translator = translate_sp()
            elif type == "mecab":
                translator = translate_mecab()
            for s in example_sentence:
                translator.translate(
                    s,
                    model,
                    enc_tokenizer,
                    dec_tokenizer,
                    show_translate=show_translate,
                    plot_attention=plot_attention,
                )


# 번역 생성 함수
# sentence piece
class translate_sp:
    def translate(
        self, sentence, model, src_tokenizer, tgt_tokenizer, plot_attention=False
    ):
        pieces, result, enc_attns, dec_attns, dec_enc_attns = self.evaluate(
            sentence, model, src_tokenizer, tgt_tokenizer, len(sentence)
        )

        print("Input: %s" % (sentence))
        print("Predicted translation: {}".format(result))

        if plot_attention:
            visualize_attention(
                pieces, result.split(), enc_attns, dec_attns, dec_enc_attns
            )

    def evaluate(self, sentence, model, src_tokenizer, tgt_tokenizer, max_men):
        sentence = preprocess_sentence(sentence)

        pieces = src_tokenizer.encode_as_pieces(sentence)
        tokens = src_tokenizer.encode_as_ids(sentence)

        _input = tf.keras.preprocessing.sequence.pad_sequences(
            [tokens], maxlen=max_men, padding="post"
        )

        ids = []
        output = tf.expand_dims([tgt_tokenizer.bos_id()], 0)
        for i in range(max_men):
            enc_padding_mask, combined_mask, dec_padding_mask = generate_masks(
                _input, output
            )

            predictions, enc_attns, dec_attns, dec_enc_attns = model(
                _input, output, enc_padding_mask, combined_mask, dec_padding_mask
            )

            predicted_id = (
                tf.argmax(tf.math.softmax(predictions, axis=-1)[0, -1]).numpy().item()
            )

            if tgt_tokenizer.eos_id() == predicted_id:
                result = tgt_tokenizer.decode_ids(ids)
                return pieces, result, enc_attns, dec_attns, dec_enc_attns

            ids.append(predicted_id)
            output = tf.concat([output, tf.expand_dims([predicted_id], 0)], axis=-1)

        result = tgt_tokenizer.decode_ids(ids)

        return pieces, result, enc_attns, dec_attns, dec_enc_attns


class translate_mecab:
    def translate(
        self,
        sentence,
        model,
        src_tokenizer,
        tgt_tokenizer,
        show_translate=False,
        plot_attention=False,
    ):
        pieces, result, enc_attns, dec_attns, dec_enc_attns = self.evaluate(
            sentence, model, src_tokenizer, tgt_tokenizer, len(sentence)
        )
        result = "".join(result)
        if show_translate:
            print("Input: %s" % (sentence))
            print("Predicted translation: {}".format(result))

        if plot_attention:
            visualize_attention(
                pieces, result.split(), enc_attns, dec_attns, dec_enc_attns
            )
        return result

    def evaluate(self, sentence, model, src_tokenizer, tgt_tokenizer, max_men):
        from konlpy.tag import Mecab

        mecab = Mecab(dicpath=r"C:\mecab\share\mecab-ko-dic")
        sentence = preprocess_sentence(sentence)
        pieces = mecab.morphs(sentence)
        tokens = src_tokenizer.texts_to_sequences([pieces])

        _input = tf.keras.preprocessing.sequence.pad_sequences(tokens, padding="post")
        ids = []
        output = tf.expand_dims([tgt_tokenizer.word_index["<start>"]], 0)
        for i in range(max_men):
            enc_padding_mask, combined_mask, dec_padding_mask = generate_masks(
                _input, output
            )

            predictions, enc_attns, dec_attns, dec_enc_attns = model(
                _input, output, enc_padding_mask, combined_mask, dec_padding_mask
            )

            predicted_id = (
                tf.argmax(tf.math.softmax(predictions, axis=-1)[0, -1]).numpy().item()
            )

            ids.append(predicted_id)
            if tgt_tokenizer.word_index["<end>"] == predicted_id:
                result = tgt_tokenizer.sequences_to_texts([ids])
                return pieces, result, enc_attns, dec_attns, dec_enc_attns

            output = tf.concat([output, tf.expand_dims([predicted_id], 0)], axis=-1)

        result = tgt_tokenizer.sequences_to_texts([ids])

        return pieces, result, enc_attns, dec_attns, dec_enc_attns


class translate:

    def translate(
        self,
        sentence,
        model,
        src_tokenizer,
        tgt_tokenizer,
        src_tokenizer_type,
        tgt_tokenizer_type,
        show_translate=False,
        plot_attention=False,
    ):
        pieces, result, enc_attns, dec_attns, dec_enc_attns = self.evaluate(
            sentence,
            model,
            src_tokenizer,
            tgt_tokenizer,
            len(sentence),
            src_tokenizer_type,
            tgt_tokenizer_type,
        )
        result = "".join(result)
        if show_translate:
            print("Input: %s" % (sentence))
            print("Predicted translation: {}".format(result))

        if plot_attention:
            visualize_attention(
                pieces, result.split(), enc_attns, dec_attns, dec_enc_attns
            )
        return result

    def evaluate(
        self,
        sentence,
        model,
        src_tokenizer,
        tgt_tokenizer,
        max_men,
        src_tokenizer_type,
        tgt_tokenizer_type,
    ):
        sentence = preprocess_sentence(sentence)

        if src_tokenizer_type == "mecab":
            from konlpy.tag import Mecab

            mecab = Mecab(dicpath=r"C:\mecab\share\mecab-ko-dic")
            pieces = mecab.morphs(sentence)
            tokens = src_tokenizer.texts_to_sequences([pieces])
        elif src_tokenizer_type == "sp":
            pieces = [src_tokenizer.encode_as_pieces(sentence)]
            tokens = [src_tokenizer.encode_as_ids(sentence)]

        _input = tf.keras.preprocessing.sequence.pad_sequences(tokens, padding="post")

        ids = []
        if tgt_tokenizer_type == "mecab":
            tgt_end_token_id = tgt_tokenizer.word_index["<end>"]
            tgt_start_token_id = tgt_tokenizer.word_index["<start>"]
            sequences_to_texts = tgt_tokenizer.sequences_to_texts()

        elif tgt_tokenizer_type == "sp":
            tgt_end_token_id = tgt_tokenizer.eos_id()
            tgt_start_token_id = tgt_tokenizer.bos_id()
            sequences_to_texts = tgt_tokenizer.decode_ids()

        output = tf.expand_dims([tgt_start_token_id], 0)

        for i in range(max_men):
            enc_padding_mask, combined_mask, dec_padding_mask = generate_masks(
                _input, output
            )

            predictions, enc_attns, dec_attns, dec_enc_attns = model(
                _input, output, enc_padding_mask, combined_mask, dec_padding_mask
            )

            predicted_id = (
                tf.argmax(tf.math.softmax(predictions, axis=-1)[0, -1]).numpy().item()
            )

            ids.append(predicted_id)
            if predicted_id == tgt_end_token_id:

                result = sequences_to_texts([ids])
                return pieces, result, enc_attns, dec_attns, dec_enc_attns

            output = tf.concat([output, tf.expand_dims([predicted_id], 0)], axis=-1)

        result = sequences_to_texts([ids])

        return pieces, result, enc_attns, dec_attns, dec_enc_attns


### 평가 및 시각화
# 레이어마다 그리기
def visualize_attention(src, tgt, enc_attns, dec_attns, dec_enc_attns):
    def draw(data, ax, x="auto", y="auto"):
        import seaborn

        seaborn.heatmap(
            data,
            square=True,
            vmin=0.0,
            vmax=1.0,
            cbar=False,
            ax=ax,
            xticklabels=x,
            yticklabels=y,
        )

    for layer in range(0, 2, 1):
        fig, axs = plt.subplots(1, 4, figsize=(20, 10))
        print("Encoder Layer", layer + 1)
        for h in range(4):
            draw(enc_attns[layer][0, h, : len(src), : len(src)], axs[h], src, src)
        plt.show()

    for layer in range(0, 2, 1):
        fig, axs = plt.subplots(1, 4, figsize=(20, 10))
        print("Decoder Self Layer", layer + 1)
        for h in range(4):
            draw(dec_attns[layer][0, h, : len(tgt), : len(tgt)], axs[h], tgt, tgt)
        plt.show()

        print("Decoder Src Layer", layer + 1)
        fig, axs = plt.subplots(1, 4, figsize=(20, 10))
        for h in range(4):
            draw(dec_enc_attns[layer][0, h, : len(tgt), : len(src)], axs[h], src, tgt)
        plt.show()


# 어텐션 그리기
def plot_attention(attention, sentence, predicted_sentence):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(1, 1, 1)
    ax.matshow(attention, cmap="viridis")

    fontdict = {"fontsize": 14}

    ax.set_xticklabels([""] + sentence, fontdict=fontdict, rotation=90)
    ax.set_yticklabels([""] + predicted_sentence, fontdict=fontdict)

    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.show()
