package chris;

import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.Random;

import moa.streams.InstanceStream;
import moa.streams.generators.RandomTreeGenerator;

import org.apache.commons.math3.distribution.GammaDistribution;

import weka.core.Instance;
import weka.core.Utils;

public class ClassDistributionDriftGenerator {
	
	
	private InstanceStream m_instanceStream = null;
	private int m_maxInstances = 0;
	private int m_maxNumDrifts = 0;
	private GammaDistribution g = new GammaDistribution(1,1);
	private Random m_rnd = new Random(0);
	
	private int[] getClassDist(ArrayList<Instance> arr) {
		int[] counts = new int[m_instanceStream.getHeader().numClasses()];
		for(int x = 0; x < arr.size(); x++) {
			counts[ (int)arr.get(x).classValue() ] += 1;
		}
		return counts;
	}
	
	public ClassDistributionDriftGenerator(InstanceStream instanceStream, int maxInstances, int numDrifts) {
		m_instanceStream = instanceStream;
		m_maxInstances = maxInstances;
		m_maxNumDrifts = numDrifts;		
		g.reseedRandomGenerator(0);
	}
	
	public void generate() {
		
		int numClasses = m_instanceStream.getHeader().numClasses();
		
		System.out.println(m_instanceStream.getHeader());
		
		for(int iter = 0; iter < m_maxNumDrifts; iter++) {	
			
			int m = m_maxInstances / m_maxNumDrifts;
			
			double[] dirichletSample = new double[numClasses];
			for(int k = 0; k < dirichletSample.length; k++) {
				dirichletSample[k] = g.sample();
			}
			Utils.normalize(dirichletSample);
			int[] props = new int[ dirichletSample.length ];
			for(int k = 0; k < dirichletSample.length; k++) {
				props[k] = (int)(dirichletSample[k] * m );
			}
			System.err.println("want this distribution: " + Arrays.toString(props));
			
			// start sampling instances
			HashMap<Integer, ArrayDeque<Instance> > arr = new HashMap<Integer, ArrayDeque<Instance>>();
			for(int k = 0; k < numClasses; k++) arr.put(k, new ArrayDeque<Instance>());
			for(int x = 0; x < m; x++) {
				Instance inst = m_instanceStream.nextInstance();
				arr.get((int)inst.classValue()).push(inst);
			}
			
			ArrayList<Instance> tmp = new ArrayList<Instance>();
			
			// ok, now grab the right amounts
			for(int k = 0; k < arr.size(); k++) {
				ArrayDeque<Instance> stack = arr.get(k);
				for(int z = 0; z < props[k]; z++) {
					if(stack.size() != 0) {
						//System.out.println(stack.pop());
						tmp.add( stack.pop() );
					}
				}
			}
			
			Collections.shuffle(tmp);
			for(Instance inst : tmp) {
				System.out.println(inst);
			}
			
			
		}
		
	}

}
