package moa.classifiers.functions;

import moa.classifiers.AbstractClassifier;
import moa.core.Measurement;
import moa.options.FlagOption;
import moa.options.FloatOption;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.SparseInstance;
import weka.core.Utils;

/**
 * Exponential moving average algorithm. This implementation
 * is for sparse instances, and should be much faster than
 * the dense implementation.
 * @author cjb60
 *
 */
public class SparseEMA extends AbstractClassifier {

	private static final long serialVersionUID = -4357871674824496401L;

	private double[][] m_weightMatrix = null;
	
	public FloatOption learningRateOption = new FloatOption("learningRate",
			'b', "Learning rate", 0.15, 0, 1);
	public FloatOption updateMarginOption = new FloatOption("updateMargin",
			'm', "Max margin", 0.15, 0, 1);
	public FlagOption dontUseWeightsOption = new FlagOption("dontUseWeights",
			'd', "Don't use weights");
	
	private double beta = learningRateOption.getValue();
	private double dm = updateMarginOption.getValue();
	private double wmin = 0.01;
	
	private boolean m_d = false;
	
	// if dx < dm then update
	// the higher the margin (e.g. from 0.15 to 0.5)
	// the more updating that occurs
	public double getUpdateMargin() {
		return dm;
	}
	
	public void setUpdateMargin(double d) {
		dm = d;
	}
	
	// also boost amount
	public double getLearningRate() {
		return beta;
	}
	
	public void setLearningRate(double b) {
		beta = b;
	}
	
	public double[][] getWeightMatrix() {
		return m_weightMatrix;
	}

	@Override
	public boolean isRandomizable() {
		return false;
	}

	@Override
	public double[] getVotesForInstance(Instance instance) {
		if( m_weightMatrix == null ) {
			m_weightMatrix = new double[instance.numAttributes()-1][instance.numClasses()];
		}
		
		// score the connected classes
		double[] scores = new double[instance.numClasses()];
		for(int c = 0; c < instance.numClasses(); c++) {
			// iterate through the instance x
			double sum = 0;
			for(int f = 0; f < instance.numValues(); f++) {
				if( instance.index(f) == instance.classIndex() ) {
					continue;
				}
				// todo
				sum += (instance.valueSparse(f) * m_weightMatrix[ instance.index(f) ][c]);
			}
			scores[c] = sum;
		}
		try {
			Utils.normalize(scores);
		} catch(Exception ex) {
			return new double[ instance.numClasses() ];
		}
		return scores;
	}

	@Override
	public void resetLearningImpl() {
		m_weightMatrix = null;
		setLearningRate( learningRateOption.getValue() );
		setUpdateMargin( updateMarginOption.getValue() );
		m_d = dontUseWeightsOption.isSet();
	}
	
	@Override
	public void trainOnInstanceImpl(Instance inst) {
		if( m_weightMatrix == null ) {
			m_weightMatrix = new double[inst.numAttributes()-1][inst.numClasses()];
		}
		
		int numIterations = (int) inst.weight();
		if(m_d) numIterations = 1;
		for(int z = 0; z < numIterations; z++) {
		
			// score the connected classes
			double[] scores = new double[inst.numClasses()];
			for(int c = 0; c < inst.numClasses(); c++) {
				// iterate through the instance x
				double sum = 0;
				for(int f = 0; f < inst.numValues(); f++) {
					if( inst.index(f) == inst.classIndex() ) {
						continue;
					}
					sum += (inst.valueSparse(f) * m_weightMatrix[ inst.index(f) ][c]);
				}
				scores[c] = sum;
			}
			
			// ok, get the score of the actual class, and the score
			// of the most common class that's not the actual class
			double scoreOfActualClass = scores[ (int)inst.classValue() ];
			double scorePrime = scores[0];
			int[] sortedIndices = Utils.sort(scores);
			for(int i = sortedIndices.length-1; i >= 0; i--) {
				if( sortedIndices[i] != (int)inst.classValue() ) {
					scorePrime = scores[ sortedIndices[i] ];
					break;
				}
			}
			
			// compute the margin
			// if margin is not met, then update
			double dx = scoreOfActualClass - scorePrime;	
			if( dx < dm ) {
				// for all f in x, do...
				for(int f = 0; f < inst.numValues(); f++) {
					if( inst.index(f) == inst.classIndex() ) {
						continue;
					}
					// all active features' connections are first decayed,
					// then the connections to the true class are boosted
					// 3.1
					for(int c = 0; c < inst.numClasses(); c++) {
						//if( c == x.classValue() ) continue;
						// (1 - x_f^2 * beta) * w_fc
						m_weightMatrix[ inst.index(f) ][c] = ( 1 - (inst.valueSparse(f)*inst.valueSparse(f)*beta) ) *
								m_weightMatrix[ inst.index(f) ][c];
					}
					// 3.2
					// boost connection to the true class
					m_weightMatrix[ inst.index(f) ][(int)inst.classValue()] += (inst.valueSparse(f)*beta);
					// 3.3
					// then, drop any tiny weights
					for(int c = 0; c < inst.numClasses(); c++) {
						if( m_weightMatrix[ inst.index(f) ][c] < wmin) {
							m_weightMatrix[ inst.index(f) ][c] = 0;
						}
					}
				}
			}
		
		}
		
	}

	@Override
	protected Measurement[] getModelMeasurementsImpl() {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public void getModelDescription(StringBuilder out, int indent) {
		// TODO Auto-generated method stub
		
	}

}
