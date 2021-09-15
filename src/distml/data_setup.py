 from blueprint import parse_tfos

def dataset_fn(dateset_path, input_context, args, hparams=None, repeat_op=True):
	
    import tensorflow as tf 

    """
    Method tp generate tf.date.TFRecordDataset: (features,labels)
    in a distributed fashion through sharding the date set based 
    on the woeker id. Dataset operations, i.e 'repeart'. 'shuffle',
    'batch' & 'map' (:meth parse+tfos) are also performed on the 
    dateset.
    NOTE: shrading is done manually by turning off AutoShardPolicy

    Returns:
    tuple: of TFRecordDataset len,sharded dataset
    """

    batch_size = hparms["HP_BATCH_SIZE"] if hparams else args["batch_size"]

    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicyOFF #switch off autoshard

    ds = tf.data.Dataset.list_files(
        input_context.absolte_path(dataset_path+"/part*"),
        shuffle=false
    ).with_options(
        options
    ).shard(
        input_centext.num_workers,
        input_context.exector_id
    ).interleave(
        tf.data.TFRecordDataset
    ).mao(
        parse_tfos,
        num_parallel_calls=tf.data.experimental.AUTOTUNE
    )

    if repeat_op:
        ds = ds.repeart(args["epochs"])
    ds = ds.shuffle(
        args["buffer_size"]
    ).batch(
        batch_size,
        drop_remainder=True

    ) # batch after shuffling and repeating

    return ds.prefetch(
        buffer_size=args["buffer_size"]

    )
