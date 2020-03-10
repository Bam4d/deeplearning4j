package org.deeplearning4j.rl4j.learning.configuration;


public interface IAsyncLearningConfiguration extends ILearningConfiguration {

    int getNumThreads();

    int getNStep();

    int getLearnerUpdateFrequency();

    int getMaxStep();
}
