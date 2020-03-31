package org.deeplearning4j.rl4j.learning.async.a3c.discrete;

import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.rl4j.experience.StateActionPair;
import org.deeplearning4j.rl4j.learning.async.AsyncGlobal;
import org.deeplearning4j.rl4j.mdp.MDP;
import org.deeplearning4j.rl4j.network.NeuralNet;
import org.deeplearning4j.rl4j.network.ac.IActorCritic;
import org.deeplearning4j.rl4j.observation.Observation;
import org.deeplearning4j.rl4j.space.ActionSpace;
import org.deeplearning4j.rl4j.space.Box;
import org.deeplearning4j.rl4j.space.ObservationSpace;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.mockito.ArgumentCaptor;
import org.mockito.Mock;
import org.mockito.junit.MockitoJUnitRunner;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.ArrayList;
import java.util.List;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNull;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

@RunWith(MockitoJUnitRunner.class)
public class AdvantageActorCriticUpdateAlgorithmTest {

    @Mock
    AsyncGlobal<NeuralNet> mockAsyncGlobal;

    @Mock
    IActorCritic mockActorCritic;

    @Test
    public void refac_calcGradient_non_terminal() {
        // Arrange
        int[] observationShape = new int[] {5};
        double gamma = 0.9;
        AdvantageActorCriticUpdateAlgorithm algorithm = new AdvantageActorCriticUpdateAlgorithm(false, observationShape, 1, gamma);

        INDArray[] originalObservations = new INDArray[] {
                Nd4j.create(new double[] { 0.0, 0.1, 0.2, 0.3, 0.4 }),
                Nd4j.create(new double[] { 1.0, 1.1, 1.2, 1.3, 1.4 }),
                Nd4j.create(new double[] { 2.0, 2.1, 2.2, 2.3, 2.4 }),
                Nd4j.create(new double[] { 3.0, 3.1, 3.2, 3.3, 3.4 }),
        };

        int[] actions = new int[] { 0, 1, 2, 1 };
        double[] rewards = new double[] { 0.1, 1.0, 10.0, 100.0 };

        List<StateActionPair<Integer>> experience = new ArrayList<StateActionPair<Integer>>();
        for(int i = 0; i < originalObservations.length; ++i) {
            experience.add(new StateActionPair<>(new Observation(originalObservations[i]), actions[i], rewards[i], false));
        }

        when(mockActorCritic.outputAll(any(INDArray.class))).thenAnswer(invocation -> {
            INDArray batch = invocation.getArgument(0);
            return new INDArray[] { batch.mul(-1.0) };
        });

        ArgumentCaptor<INDArray> inputArgumentCaptor = ArgumentCaptor.forClass(INDArray.class);
        ArgumentCaptor<INDArray[]> criticActorArgumentCaptor = ArgumentCaptor.forClass(INDArray[].class);

        // Act
        algorithm.computeGradients(mockActorCritic, experience);

        verify(mockActorCritic, times(1)).gradient(inputArgumentCaptor.capture(), criticActorArgumentCaptor.capture());

        assertEquals(Nd4j.zeros(1), inputArgumentCaptor.getValue());
        assertEquals(Nd4j.zeros(1), criticActorArgumentCaptor.getValue()[0]);
        assertEquals(Nd4j.zeros(1), criticActorArgumentCaptor.getValue()[1]);

    }

}
