import matplotlib.pyplot as plt
import numpy as np
import sys
from pylab import plot, show, savefig, xlim, figure, \
                hold, ylim, legend, boxplot, setp, axes

# function for setting the colors of the box plots pairs
def setBoxColors(bp):
    setp(bp['boxes'][0], color='blue')
    setp(bp['caps'][0], color='blue')
    setp(bp['caps'][1], color='blue')
    setp(bp['whiskers'][0], color='blue')
    setp(bp['whiskers'][1], color='blue')
    setp(bp['fliers'][0], color='blue')
    setp(bp['fliers'][1], color='blue')
    setp(bp['medians'][0], color='blue')

    setp(bp['boxes'][1], color='red')
    setp(bp['caps'][2], color='red')
    setp(bp['caps'][3], color='red')
    setp(bp['whiskers'][2], color='red')
    setp(bp['whiskers'][3], color='red')
    setp(bp['fliers'][2], color='red')
    setp(bp['fliers'][3], color='red')
    setp(bp['medians'][1], color='red')



if __name__ == '__main__':

    MAX_CHAR_NUM = 30

    # read HMM accuracy data
    hmm_data = [[] for i in range(30)] # assume mostly 30 characters in a movie
    f = open('./../data/c_hmm_1-200.txt', 'r')
    for l in f:
        d = l.split()
        # print d
        num_char = int(d[0])
        accur = float(d[1])
        if num_char <= MAX_CHAR_NUM:
            hmm_data[num_char - 1].append(accur)
        else:
            print "num_char = %d, accuracy = %f"%(num_char, accur)
    f.close()

    # read Naive Bayes accuracy data
    nb_data = [[] for i in range(30)] # assume mostly 30 characters in a movie
    f = open('./../data/c_nb_1-200.txt', 'r')
    for l in f:
        d = l.split()
        # print d
        num_char = int(d[0])
        accur = float(d[1])
        if num_char <= MAX_CHAR_NUM:
            nb_data[num_char - 1].append(accur)
        else:
            print "num_char = %d, accuracy = %f"%(num_char, accur)
    f.close()

    # generate random guess baseline
    x = [2*i+1.5 for i in range(MAX_CHAR_NUM)]
    y = [1.0/float(i+1) for i in range(MAX_CHAR_NUM)]


    # start plot
    fig = figure()
    ax = axes()
    hold(True)
    for i in range(MAX_CHAR_NUM):
        data = [nb_data[i], hmm_data[i]]
        print data
        if (len(data[0]) > 0 and len(data[1]) > 0):
            bp = boxplot(data,
                         positions = [2*i+1, 2*i+2],
                         widths = 0.8)
            # print type(bp)
            # print len(bp)
            # print bp['boxes']
            setBoxColors(bp)

    # set axes limits and labels
    xlim(0, 1)
    ylim(0, 0.7)
    ax.set_xticklabels([str(i+1) for i in range(MAX_CHAR_NUM)])
    ax.set_xticks([2*i + 1.5 for i in range(MAX_CHAR_NUM)])

    # draw temporary red and blue lines and use them to create a legend
    hB, = plot([1,1],'b-')
    hR, = plot([1,1],'r-')
    baseline, = plt.plot(x, y, 'k--', label='Random Guess')

    print type(hB)
    print hB
    print type(baseline)
    print baseline

    legend((hB, hR, baseline),('Naive Bayes', 'HMM', 'Random Guess'))
    # legend = ax.legend(loc='upper center')
    # plt.legend([][])
    hB.set_visible(False)
    hR.set_visible(False)

    plt.xlabel('Number of characters')
    plt.ylabel('Accuracy of classifing character from line')

    # savefig('boxcompare.png')
    show()

    # plt.boxplot([hmm_data], [nb_data], 0, 'gD')
    # plt.show()
    sys.exit()



    # fake up some data
    spread = np.random.rand(50) * 100
    center = np.ones(25) * 50
    flier_high = np.random.rand(10) * 100 + 100
    flier_low = np.random.rand(10) * -100
    data = np.concatenate((spread, center, flier_high, flier_low), 0)

    # basic plot
    # print data
    # plt.boxplot(data)
    #
    # # notched plot
    # plt.figure()
    # plt.boxplot(data, 1)
    #
    # # change outlier point symbols
    # plt.figure()
    # plt.boxplot(data, 0, 'gD')
    #
    # # don't show outlier points
    # plt.figure()
    # plt.boxplot(data, 0, '')
    #
    # # horizontal boxes
    # plt.figure()
    # plt.boxplot(data, 0, 'rs', 0)
    #
    # # change whisker length
    # plt.figure()
    # plt.boxplot(data, 0, 'rs', 0, 0.75)

    # fake up some more data
    spread = np.random.rand(50) * 100
    center = np.ones(25) * 40
    flier_high = np.random.rand(10) * 100 + 100
    flier_low = np.random.rand(10) * -100
    d2 = np.concatenate((spread, center, flier_high, flier_low), 0)
    data.shape = (-1, 1)
    d2.shape = (-1, 1)
    # data = concatenate( (data, d2), 1 )
    # Making a 2-D array only works if all the columns are the
    # same length.  If they are not, then use a list instead.
    # This is actually more efficient because boxplot converts
    # a 2-D array into a list of vectors internally anyway.
    data = [data, d2, d2[::2, 0]]
    print data
    print type(data)
    print len(data)
    print data[0]
    print type(data[0])
    print len(data[0])
    print data[0][0]

    # multiple box plots on one figure
    plt.figure()
    plt.boxplot(data)

    plt.show()
