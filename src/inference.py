"""
Inference: greedy autoregressive decoding.

During inference, the decoder generates tokens one at a time, using previously
generated tokens as input. This is in contrast to training, where ground truth
targets are provided (teacher forcing).
"""

import tensorflow as tf
from .masks import create_padding_mask, create_masks


def greedy_decode(model, src_seq, start_token=1, end_token=2, max_len=16):
    """
    Greedy autoregressive decoding: always select highest-probability token.
    
    Process:
    1. Encode source sequence once
    2. Initialize with start token
    3. Iteratively: predict next token, append to sequence, repeat until end token
    
    Args:
        model: Transformer model
        src_seq: source sequence token IDs (src_len,)
        start_token: token ID marking sequence start
        end_token: token ID marking sequence end
        max_len: maximum generation length
    
    Returns:
        List of generated token IDs
    """
    # Prepare source: add batch dimension
    src = tf.constant(src_seq)[tf.newaxis, :]  # (1, src_len)
    enc_padding_mask = create_padding_mask(src)
    
    # Encode source once
    enc_output = model.encoder(src, mask=enc_padding_mask, training=False)

    # Initialize with start token
    output = tf.constant([[start_token]], dtype=tf.int32)  # (1, 1)
    
    # Autoregressive generation loop
    for _ in range(max_len):
        # Create masks for current output sequence
        _, look_ahead_mask, dec_padding_mask = create_masks(src, output)
        
        # Decode: get logits for next token
        logits = model.decoder(output, enc_output, 
                              look_ahead_mask=look_ahead_mask, 
                              padding_mask=dec_padding_mask, 
                              training=False)
        
        # Greedy selection: take token with highest probability
        next_token_logits = logits[:, -1, :]  # (1, vocab_size)
        next_token_id = tf.argmax(next_token_logits, axis=-1, output_type=tf.int32)  # (1,)
        
        # Append to output sequence
        output = tf.concat([output, tf.expand_dims(next_token_id, axis=1)], axis=1)
        
        # Stop if end token generated
        if int(next_token_id.numpy()[0]) == end_token:
            break
    
    return output.numpy().tolist()[0]
