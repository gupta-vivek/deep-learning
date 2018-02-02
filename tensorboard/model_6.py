from tensorflow.python.summary import event_accumulator as ea

acc_tf = ea.EventAccumulator("/tmp/test/1/events.out.tfevents.1493210988.rztl1005")
acc_tf.Reload()

path = "/tmp/test/2/events.out.tfevents.1493211083.rztl1005"
acc_rzt = ea.EventAccumulator(path)
acc_rzt.Reload()

print(acc_tf.Tags())
print(acc_rzt.Tags())
