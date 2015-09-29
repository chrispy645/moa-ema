package moa.classifiers.meta;

import org.apache.commons.math3.distribution.GammaDistribution;

import moa.classifiers.Classifier;
import moa.classifiers.core.driftdetection.ADWIN;
import moa.core.MiscUtils;
import weka.core.Instance;

public class LeveragingBagGamma extends LeveragingBag {
	
	private static final long serialVersionUID = 8603087203589227047L;
	
	private transient GammaDistribution m_gamma = null;
	
    @Override
    public void resetLearningImpl() {
    	m_gamma = new GammaDistribution(1,1);
    	m_gamma.reseedRandomGenerator(this.classifierRandom.nextLong());
    	
        this.ensemble = new Classifier[this.ensembleSizeOption.getValue()];
        Classifier baseLearner = (Classifier) getPreparedClassOption(this.baseLearnerOption);
        baseLearner.resetLearning();
        for (int i = 0; i < this.ensemble.length; i++) {
            this.ensemble[i] = baseLearner.copy();
        }
        this.ADError = new ADWIN[this.ensemble.length];
        for (int i = 0; i < this.ensemble.length; i++) {
            this.ADError[i] = new ADWIN((double) this.deltaAdwinOption.getValue());
        }
        this.numberOfChangesDetected = 0;
        if (this.outputCodesOption.isSet()) {
            this.initMatrixCodes = true;
        }
    }
	
    @Override
    public void trainOnInstanceImpl(Instance inst) {
        int numClasses = inst.numClasses();
        //Output Codes
        if (this.initMatrixCodes == true) {
            this.matrixCodes = new int[this.ensemble.length][inst.numClasses()];
            for (int i = 0; i < this.ensemble.length; i++) {
                int numberOnes;
                int numberZeros;

                do { // until we have the same number of zeros and ones
                    numberOnes = 0;
                    numberZeros = 0;
                    for (int j = 0; j < numClasses; j++) {
                        int result = 0;
                        if (j == 1 && numClasses == 2) {
                            result = 1 - this.matrixCodes[i][0];
                        } else {
                            result = (this.classifierRandom.nextBoolean() ? 1 : 0);
                        }
                        this.matrixCodes[i][j] = result;
                        if (result == 1) {
                            numberOnes++;
                        } else {
                            numberZeros++;
                        }
                    }
                } while ((numberOnes - numberZeros) * (numberOnes - numberZeros) > (this.ensemble.length % 2));

            }
            this.initMatrixCodes = false;
        }


        boolean Change = false;
        Instance weightedInst = (Instance) inst.copy();
        double w = this.weightShrinkOption.getValue();

        //Train ensemble of classifiers
        for (int i = 0; i < this.ensemble.length; i++) {
            double k = 0.0;
            switch (this.leveraginBagAlgorithmOption.getChosenIndex()) {
                case 0: //LeveragingBag
                	k = m_gamma.sample();
                    //k = MiscUtils.poisson(w, this.classifierRandom);
                    break;
                case 1: //LeveragingBagME
                    double error = this.ADError[i].getEstimation();
                    k = !this.ensemble[i].correctlyClassifies(weightedInst) ? 1.0 : (this.classifierRandom.nextDouble() < (error / (1.0 - error)) ? 1.0 : 0.0);
                    break;
                case 2: //LeveragingBagHalf
                    w = 1.0;
                    k = this.classifierRandom.nextBoolean() ? 0.0 : w;
                    break;
                case 3: //LeveragingBagWT
                    w = 1.0;
                    //k = 1.0 + MiscUtils.poisson(w, this.classifierRandom);
                    k = 1.0 + m_gamma.sample();
                    break;
                case 4: //LeveragingSubag
                    w = 1.0;
                    //k = MiscUtils.poisson(1, this.classifierRandom);
                    k = m_gamma.sample();
                    k = (k > 0) ? w : 0;
                    break;
            }
            if (k > 0) {
                if (this.outputCodesOption.isSet()) {
                    weightedInst.setClassValue((double) this.matrixCodes[i][(int) inst.classValue()]);
                }
                weightedInst.setWeight(inst.weight() * k);
                this.ensemble[i].trainOnInstance(weightedInst);
            }
            boolean correctlyClassifies = this.ensemble[i].correctlyClassifies(weightedInst);
            double ErrEstim = this.ADError[i].getEstimation();
            if (this.ADError[i].setInput(correctlyClassifies ? 0 : 1)) {
                if (this.ADError[i].getEstimation() > ErrEstim) {
                    Change = true;
                }
            }
        }
        if (Change) {
            numberOfChangesDetected++;
            double max = 0.0;
            int imax = -1;
            for (int i = 0; i < this.ensemble.length; i++) {
                if (max < this.ADError[i].getEstimation()) {
                    max = this.ADError[i].getEstimation();
                    imax = i;
                }
            }
            if (imax != -1) {
                this.ensemble[imax].resetLearning();
                //this.ensemble[imax].trainOnInstance(inst);
                this.ADError[imax] = new ADWIN((double) this.deltaAdwinOption.getValue());
            }
        }
    }


}
