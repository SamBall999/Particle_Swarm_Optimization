import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mannwhitneyu
import pandas as pd
import numpy as np
#from custom_id3 import ID3_Decision_Tree
#from testing import calculate_rates, accuracy, confusion_matrix, one_sided_mann_whitney, two_sided_mann_whitney


def one_sided_mann_whitney(tabu_data, ga_data):

    """
    Performs one sided statistical hypothesis test for two sets of independent samples.

    Arguments:
    - Test accuracies obtained from feature subset selected using tabu search
    - Test accuracies obtained from feature subset selected using genetic algorithm

    Returns:
    - Statistic and p value from one-sided Mann-Whitney test.
    """

    stat, p = mannwhitneyu(tabu_data, ga_data)
    print("U Statistic: {}".format(stat))
    print("p-value: {}".format(p))

    return stat, p




def two_sided_mann_whitney(tabu_data, ga_data):

    """
    Performs two sided statistical hypothesis test for two sets of independent samples.

    Arguments:
    - Test accuracies obtained from feature subset selected using tabu search
    - Test accuracies obtained from feature subset selected using genetic algorithm

    Returns:
    - Statistic and p value from Mann-Whitney test.
    """

    stat, p = mannwhitneyu(tabu_data, ga_data, alternative='two-sided')
    print("U Statistic: {}".format(stat))
    print("p-value: {}".format(p))

    return stat, p







def plot_performance_dist(data, c, alg_label, fig_name):
    
    #ax = sns.distplot(tabu_data, color = "cornflowerblue", label = "Tabu Search")
    ax = sns.distplot(data, color = c, label = alg_label)
    plt.xlabel("Test Accuracy")
    plt.ylabel("Density")
    plt.savefig(fig_name, dpi = 300)
    plt.show()


def plot_all(data_1, data_2, c_1, c_2, alg_label_1,  alg_label_2, fig_name):
    
    ax = sns.distplot(data_1, color = c_1, label = alg_label_1)
    plt.xlabel("Test Accuracy")
    plt.ylabel("Density")
    ax = sns.distplot(data_2, color = c_2, label = alg_label_2)
    plt.legend()
    plt.savefig(fig_name, dpi = 300)
    plt.show()


def plot_confusion_matrix(array, colourmap):

    df_cm = pd.DataFrame(array, index=["True", "False"],  columns=["True", "False"])
    sns.set(font_scale=1.2) # for label size
    df_cm.index.name = 'Actual'
    df_cm.columns.name = 'Predicted'
    sns.heatmap(df_cm, annot=True, cmap = colourmap, annot_kws={"size": 18}, fmt="d") # font size #light=1
    plt.show()



def classification_performance(candidate_x, training_data, test_data):

    feature_indices = [(i+1) for i in range(len(candidate_x)) if candidate_x[i]==1]
    feature_list =  training_data.columns.values[feature_indices] 
  
    # convert x and y data to numpy arrays
    x = np.array(training_data.drop("Target", axis=1).copy())
    y = np.array(training_data["Target"])

    # create decision tree with given subset
    tree = ID3_Decision_Tree(x, y) # intialize decision tree
    tree.id_3(feature_list) # build tree


    # calculate accuracy on the test set
    X_test = np.array(test_data.drop("Target", axis=1).copy())
    test_predictions = tree.predict_batch(X_test)
    test_targets = test_data["Target"].values
    TP, FP, TN, FN = calculate_rates(test_targets, test_predictions) # calculate number of true positives, false positives etc.
    test_accuracy = accuracy(TP, FP, TN, FN) 
    print("Test Accuracy: {}".format(test_accuracy))
    confusion_matrix(test_targets, test_predictions)

    return TP, FP, TN, FN 



def main():

    # Insert performance data - set of test accuracies obtained by each algorithm
    #tabu_data = [0.919, 0.9095, 0.8755, 0.904, 0.9265, 0.858, 0.908, 0.87, 0.875, 0.914, 0.921, 0.907, 0.9155, 0.8755, 0.923, 0.923, 0.9145, 0.9155, 0.882, 0.926, 0.919, 0.901, 0.9265, 0.9095, 0.9165, 0.883, 0.903, 0.9165, 0.897, 0.91]
    #tabu_data = [0.919, 0.9095, 0.8755, 0.904, 0.9265, 0.858, 0.908, 0.87, 0.875, 0.914, 0.921, 0.907, 0.9155, 0.8755, 0.923]
    #ga_data = [0.871, 0.7915, 0.906, 0.877, 0.8965, 0.8865, 0.8585, 0.8605, 0.845, 0.8755, 0.908, 0.879, 0.8935, 0.839, 0.864, 0.8885, 0.8385, 0.8755, 0.8995, 0.8155, 0.858, 0.8425, 0.8815, 0.885, 0.889, 0.888, 0.8765, 0.831, 0.917, 0.862]
    #ga_data = [0.871, 0.7915, 0.906, 0.877, 0.8965, 0.8865, 0.8585, 0.8605, 0.845, 0.8755, 0.908, 0.879, 0.8935, 0.839, 0.864] # note this does not include 0.917

    ipso_data = [7.338824258442943e-25, 1.0516430878648214e-26, 1.9517598755785428e-26, 1.7822400790570798e-26, 2.3422024397683203e-26, 4.106674321649383e-26, 7.268194329602018e-27, 6.402209612145861e-26, 2.412282931934767e-28, 1.0877242951088832e-25, 9.023327287691284e-26, 2.764582805921635e-27, 4.555437101890341e-26, 6.992968853103195e-24, 1.7864079260694432e-27, 3.620961881779172e-27, 5.588383549655172e-26, 8.419928437042587e-27, 7.66226745393208e-29, 2.4265612172242195e-25, 6.144141566788568e-26, 2.42327190173975e-25, 2.2582052464541625e-23, 6.642939428984198e-25, 4.573481547843408e-28, 1.4919707281742744e-27, 6.706228967947947e-26, 2.101562076608886e-25, 9.167902278200837e-28, 5.973266690672255e-28]
    spso_data = [3.426692382612508e-24, 8.360562861971024e-22, 1.424154999997349e-23, 5.2241869663229904e-23, 2.352833173334382e-24, 3.1784986436456056e-23, 2.8538917918624287e-24, 5.679705982671679e-24, 1.0861499436856826e-22, 1.6575563987691357e-21, 1.0009688319976934e-22, 6.378225456896576e-22, 1.0034251337853042e-22, 1.2527691887754375e-21, 1.147952722459128e-22, 8.540792532769946e-24, 1.6054372396893346e-22, 3.604912628443964e-23, 2.654810160518304e-22, 6.513529433113382e-23, 1.6021929610006334e-22, 5.269613567375471e-24, 3.638365761513084e-23, 4.7894952793124507e-23, 5.465555083497373e-23, 9.233737628671043e-24, 1.423131871777613e-23, 9.99663308830269e-25, 3.280525384468297e-23, 1.0294754072889672e-22]

    ipso_ackley_lbest  = [7.58905556397238e-06, 1.0968646258380232e-05, 1.4793876620178281e-05, 3.850368103019974e-06, 5.569511337899513e-06, 2.436458985544121e-05, 4.587241292641764e-05, 7.383215184564307e-06, 1.905311223326933e-05, 5.48557311619291e-06, 0.00010861349797552933, 8.555525380149476e-06, 1.2581590923321784e-05, 6.591272019296213e-06, 6.593953149280907e-06, 6.102613735681217e-06, 7.55250188477774e-06, 7.734751517052274e-06, 5.960576416885743e-06, 1.5793835235644593e-05, 3.094883535670334e-06, 2.0726977667084867e-05, 7.035045208159119e-06, 7.315807721841594e-06, 6.307758562629218e-05, 4.312533374406513e-06, 4.254303057837916e-06, 1.557592213119463e-05, 5.590715133196866e-05, 1.015330987685914e-05]
    spso_ackley =[1.1551485027119637, 1.839993475812133, 1.2147284604857589e-06, 2.0133152362563504, 3.614911432414658e-07, 3.2509208525155486e-07, 1.1551485027100417, 1.9099623260743215e-07, 2.3168487918477614, 1.155148502709896, 2.076074672618944e-06, 2.7526090251583923e-07, 3.025169692749685e-05, 1.1720273723777521, 1.4235364879774335, 5.927463146981893e-07, 1.6462236331031943, 1.8399934758121508, 3.126847296874708, 2.1711858463397076, 0.00010969525100934518, 1.6462236331032405, 6.00784201498783e-07, 2.171185846341633, 4.2772348196606913e-07, 2.3271698568194665e-07, 2.3168487918479284, 1.6462236331036633, 1.4803608334723606e-06, 4.966508808657011e-07]


    ipso_m = [-14.32997003, -16.7928636, -16.73394189, -15.25100248, -15.61716652, -16.50259356, -15.6630851, -17.31079771, -16.90965415, -16.5282516, -13.38588755, -15.48016755, -14.9451827, -16.7845552, -15.44856395, -15.88909535, -13.73206863, -16.9012607, -13.62745059, -15.29975319, -15.17809344, -16.14817907, -16.81222729, -14.39932345, -16.12882556, -16.31283778, -16.22620483, -15.93130686, -14.03918553, -16.1189925]
    spso_m = [-14.065540969730508, -13.82982016563773, -12.724856948748053, -14.179058921076082, -10.936397442377487, -14.045186242383515, -12.71904838876435, -12.174420837920461, -15.30489332774648, -16.04343141802603, -15.10625365191058, -13.52118748048646, -15.187096972833492, -13.881928232146088, -10.032495615164317, -13.450780960383923, -13.769139931272345, -14.488886979116732, -12.054940902074184, -12.669317414776177, -13.3052893965615, -14.549735304861425, -14.138448274835723, -14.33846713062162, -14.45095384097917, -10.833347843275115, -10.380992883317097, -12.288312433950823, -10.639323855056531, -12.963750317535068]

    ipso_k = [0.004414850321357012, 0.10208516386795233, 0.0015800794944073153, 0.01321325148034199, 0.09680303870949344, 0.009232661765902192,  0.0052423078872704185, 0.06103929475718852, 0.04300343206415889, 0.005511258842850603, 8.425663341922462e-07, 0.05049506702126486, 0.033919745155511806, 0.003102749595032178, 0.0012090468574215592, 0.028005039867395963, 0.010468861395887256, 0.00623658251664639, 0.014257161766758918, 0.03632265942349311, 0.03651877220391899, 0.0029448471744442026, 0.00491104565589871, 0.027323238537808325, 0.0022849727804764297, 0.06231184156445422, 0.011761846662695308, 0.03688990822700137, 0.012875996446984235, 0.0114396395255087]
    spso_k = [0.6151844799924091, 0.429763391677673, 1.6504839054491636, 0.23990789387686226, 0.598978549278961, 0.839425333128883, 1.0245963168296268, 0.6386740625039259, 0.6656661459396073, 0.43615762139255077, 0.2220436714113798, 0.5564854197571465, 0.5100512897734741, 0.28418479603841146, 0.42068876325823235, 1.4586435602355505, 1.3184330834621403, 0.6227415777174339, 0.6878130968549934, 0.2591105167356491, 0.34741398879335333, 0.26419808850009524, 0.6169223225575636, 0.9410108250576755, 0.7663377687261556, 0.9088894765913843, 0.24014165091061165, 0.327732303823392, 0.5165268170365739, 0.6649676978902923]

    ipso_shu = [-1.9235126691004343e+22, -4.952868705056808e+21, -1.564618568565765e+21, -6.389445293460648e+20, -2.639632749748429e+22, -1.0741622594066798e+21, -2.1507092919980074e+22, -7.413516523788872e+21, -2.0026448785724135e+22, -9.146303602973442e+21, -1.0569560436704611e+21, -1.471166114196272e+21, -1.6351066248237315e+22, -4.152175058038764e+21, -4.014662922117754e+21, -3.0261007538110494e+22, -1.9325560306219643e+21, -4.107713638170314e+22, -8.573407260825096e+21, -3.10636062320645e+21, -1.9445835188257176e+22, -1.2869075927127659e+22, -3.66545387606381e+20, -3.4657272118863816e+20, -1.635106624815958e+22, -8.50888077348246e+20, -7.753555235471531e+21, -4.1922989191072653e+21, -1.2168112440865276e+22, -3.844873880647346e+22]
    spso_shu = [-3.665344285764776e+18, -1.5394955764271342e+18, -6.325167176972604e+18, -6.670166477653788e+19, -5.3725350678570025e+19, -5.423873743314125e+17, -7.119297769821314e+18, -1.0925494754795305e+20, -9.568909177213686e+18, -4.219641230134867e+16, -1.2782496657304607e+18, -1.2426989752210483e+18, -2.6219604499830226e+18, -2.298460510729955e+17, -1.5914914906634143e+18, -4.9854751702715117e+17, -3.0000989309569327e+19, -3.868051750577568e+18, -7.122134141967122e+17, -2.2954065266310836e+19, -1.7252479874346557e+17, -1.8389837449542822e+19, -7.574833943651738e+17, -2.3953712426478484e+18, -1.6402487317691658e+18, -2.700130184367936e+18, -2.8528903081462147e+17, -1.1218433170017965e+18, -7.423615219633944e+18, -1.8128309312043453e+18]
    
    ipso_shifted_ackley = [519.8204831468062, 500.0212359204349, 500.001037534198, 500.12723746759974, 501.21902032123234, 500.0008774510987, 501.1611919284347, 500.0004611971689, 500.00174538956696, 508.9233510568292, 519.806895806988, 508.0563940547275, 500.74199635479846, 501.15529840551676, 520.2803071005576, 500.01548376318124, 500.8958916079646, 502.31609853331923, 500.00665216307635, 516.8801072866245, 500.0039723527125, 507.98247312393596, 504.7940322758442, 502.05354735101787, 519.5113503113163, 504.104395893728, 501.151324402506, 502.97044344952525, 501.7073783784397, 501.5416788384437]
    spso_shifted_ackley = [500.00020989224294, 520.6696093805923, 500.0001835129461, 501.4235364883452, 500.0000697818958, 501.4235364882674, 500.00000137884257, 500.0000359085854, 500.0000204928207, 520.4630168570517, 501.8399934759185, 502.01331523660934, 500.0074706049335, 501.8399934758132, 502.17118620114604, 520.8989757163223, 502.3168487918604, 520.723623072598, 501.83999347582227, 501.4235364887255, 500.0000049570501, 500.00000246033613, 501.64622363351776, 501.64622363311713, 502.45255152159507, 500.0000958791452, 500.00000290657937, 502.1711858464375, 502.3168487918584, 501.6462236331246]

    
    ipso_sh_m = [490.31455850953154, 488.24006583616006, 489.62798932172353, 491.41389867159984, 490.6067708689312, 490.6718291636674, 486.58457974629744, 491.81723797288316, 492.3386212134751, 489.61486720230124, 490.0231997769863, 487.0495132407718, 488.2610006283612, 488.6284467765952, 492.9099961234785, 489.86401036948246, 492.43353932611615, 491.1550204965274, 491.6088037890403, 488.845983425828, 489.66136069865007, 491.71292390747226, 490.68443551429425, 489.96718910348477, 492.29797306184645, 487.43325029920874, 489.85861639562836, 490.1308294957721,  488.412183283693, 490.4963066725953]
    spso_sh_m = [494.3438568018189, 491.74469367864447, 491.4364879784941, 494.03193213678435, 494.24744894728246, 492.11081523571, 495.3548401195996, 491.31515907454065, 494.32042861957405, 494.62089788902017, 493.3974825433536, 491.1127263281143, 492.2513949195491, 493.2653909942472, 493.1795386267462, 494.1165808318031, 492.4336738978498, 493.4672940785906, 493.1169058328323,  490.56609650707077, 490.9087174053607, 493.9582297030989, 494.58251202428534, 492.39947176440404, 491.5527666104419, 491.32020599979853, 492.43729211789815, 494.0029137114095, 492.23834639355715, 495.3259397679145]
    
   
    ipso_sh_k = [500.0560633401742, 500.02198354230296, 500.1206107337001, 500.00990251499763, 500.08046488460457, 500.07617212245646, 500.028101798303, 500.0473103466412, 500.03825606711973, 500.03490101783296, 500.13658095226003, 500.0554321184851, 500.0458350934961, 500.4259829031222, 500.0357378750484, 500.0418745807993, 500.07994605065727, 500.0324277791358, 500.069357558812, 500.01931153672365, 500.072973903701, 500.1188845827947, 500.06580002077123, 500.0673707144885, 500.0371523380917, 500.11780658176093, 500.0793896064355, 500.084528269467, 500.0173050902959, 500.1322836802257]
    spso_sh_k = [500.33802119491673, 500.7800366057308, 500.51476027514497, 500.72119316341553, 501.0226090824218, 500.61869655076896, 500.62526420573886, 501.0129032439213, 500.85020942717443, 500.47710216156634, 500.2940604007266, 500.5185784492351, 500.7640098276796, 500.521646035223, 500.23218098217893, 500.8179166825804, 500.26206749362825, 500.2468496721794, 500.8192285498436, 500.6802332721198, 500.57712202548777, 500.3304083404627, 500.7947256475308, 500.24282427077253, 500.4672420259072, 500.4539069308367, 500.49756318716555, 500.83630457071695, 500.5991859040896, 500.78574311196957]

    ipso_sh_shu = [-3.353381577522754e+21, -5.131230414271472e+21, -4.393133215864038e+19, -6.850473969828822e+20, -4.094480469052006e+20, -7.892935511221068e+19, -3.433393351445321e+21, -1.2097672940098812e+19, -1.81844037088925e+21, -2.808108759135202e+20, -2.2749882630404853e+20, -7.092683964770442e+21, -9.5768898091797e+21, -3.6010710813293347e+19, -1.789394742585748e+21, -1.1420772776919563e+20, -7.087544033453257e+19, -1.2571487514996289e+20, -1.6839191980462859e+22, -1.8480761490951042e+20, -4.303453069650952e+20, -9.001503437002996e+20, -2.9442207560897372e+19, -3.844873878707167e+22, -1.7174417033988654e+21, -2.0864440255595754e+21, -9.46001312242349e+18, -3.5009130710374478e+19, -6.85706947725231e+20, -1.1487625320582411e+20]
    spso_sh_shu = [-2.452284017585068e+16, -1.6378185043944696e+18, -7.351443675677966e+18, -1.8777660595741913e+19, -8.624935677517111e+17, -8.496186968762171e+19, -6.440840132044486e+17, -9.137466222481247e+18, -2.6874522780306376e+18, -4.9850913686631635e+17, -6.211327037169735e+18, -1.6784285316081508e+19, -7.284750200828869e+17, -6.98252593340743e+17, -2.3948702903800246e+19, -4.04775070503267e+17, -1.5668971017601654e+18, -1.0215065321848282e+17, -2.1094867268426765e+18, -2.2039650926925952e+17, -6.121660253763812e+18, -6.105540427845237e+16, -1.9694770852412362e+18, -5.876523355369051e+16, -8.292519440781884e+18, -2.7753332836890394e+17, -1.6508866529856662e+19, -2.0123590798500934e+19, -2.08144498829127e+19, -1.268079668179137e+17]


 

   # Plot performance distributions
    plot_performance_dist(ipso_m, "cornflowerblue", "IPSO", "ipso_performance_distribution.png")
    plot_performance_dist(spso_m, "lightcoral", "SPSO2011", "spso2011_performance_distribution.png")
    plot_all(ipso_m, spso_m, "cornflowerblue", "lightcoral", "IPSO", "SPSO2011", "performance_distributions.png" )



    # Calculate U statistic for one sided Mann Whitney U Test       
    print("\nOne sided Mann Whitney U Test")
    mann_whitney_result, p = one_sided_mann_whitney( ipso_m, spso_m)

    # Calculate U statistic for two sided Mann Whitney U Test
    print("\nTwo sided Mann Whitney U Test")
    mann_whitney_result, p = two_sided_mann_whitney( ipso_m, spso_m)







if __name__ == "__main__":
    main()

