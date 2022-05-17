txt = 'rr'
MODEL = 'resnet'
with open('otchet.txt', 'a+') as f:
    for listitem in test_acc:
        f.write('%s\n' % listitem)

DROPOUT2 = 1

if DROPOUT2:
    print('1')