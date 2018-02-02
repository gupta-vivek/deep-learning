from tensorflow.python.summary import event_accumulator as ea
import filecmp

acc = ea.EventAccumulator("events.out.tfevents.1493188577.rztl1005")
acc.Reload()

acc_scalar = ea.EventAccumulator("events.out.tfevents.1493187705.rztl1005")
acc_scalar.Reload()
path_rzt = '/tmp/rzt_test/3/'

acc_p = ea.EventAccumulator("events.out.tfevents.1493190465.rztl1005")
acc_p.Reload()

acc_tf = ea.EventAccumulator("tf_events.out.tfevents.1493190465.rztl1005")
acc_tf.Reload()

acc_test = ea.EventAccumulator("/home/vivek/PycharmProjects/tensorboard/events.out.tfevents.1493190465.rztl1005")
acc_test.Reload()
# f1 = open("events.out.tfevents.1493103619.rztl1005", "rb")
# f2 = open("events.out.tfevents.1493103619.rztl1005", "rb")
# f1 = "events.out.tfevents.1493103619.rztl1005"
# f2 = "events.out.tfevents.1493103619.rztl1005"

# print(filecmp.cmp(f1, f2))

# print(acc.Tags())
# print(acc_scalar.Tags())
# print(acc_p.Tags())

print(acc_tf.Tags())
print(acc_p.Tags())
print(acc_tf.Scalars('c'))
print(acc_p.Scalars('c_1'))
print(acc_p.Scalars('c'))
print(acc_p.Scalars('c')[0][1])
print(acc_test.Tags())
# print(acc.RunMetadata('step1'))
# print(acc.RunMetadata('step100'))
