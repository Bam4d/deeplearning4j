package org.deeplearning4j.rl4j.learning.async;

import org.deeplearning4j.rl4j.learning.listener.TrainingListenerList;
import org.deeplearning4j.rl4j.mdp.MDP;
import org.deeplearning4j.rl4j.network.NeuralNet;
import org.deeplearning4j.rl4j.observation.Observation;
import org.deeplearning4j.rl4j.space.ActionSpace;
import org.deeplearning4j.rl4j.space.ObservationSpace;
import org.deeplearning4j.rl4j.support.MockObservationSpace;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.mockito.Mock;
import org.mockito.Mockito;
import org.mockito.junit.MockitoJUnitRunner;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

@RunWith(MockitoJUnitRunner.class)
public class AsyncThreadTest {

    @Mock
    ActionSpace<INDArray> mockActionSpace;

    @Mock
    ObservationSpace<INDArray> mockObservationSpace;

    @Mock
    AsyncConfiguration mockAsyncConfiguration;

    @Mock
    NeuralNet mockNeuralNet;

    @Mock IAsyncGlobal<NeuralNet> mockAsyncGlobal;

    @Mock
    MDP mockMDP;

    @Mock
    TrainingListenerList mockTrainingListeners;

    @Before
    public void setup() {

    }

    @Test
    public void when_newEposideStarted_expect_neuralNetworkReset() {

        // Arrange
        Observation mockObs = new Observation(Nd4j.zeros(3, 10, 10));

        when(mockObservationSpace.getShape()).thenReturn(new int[] {3,10,10});
        when(mockMDP.getObservationSpace()).thenReturn(mockObservationSpace);

        AsyncThread<INDArray, INDArray, ActionSpace<INDArray>, NeuralNet> thread = mock(AsyncThread.class, Mockito.withSettings()
                .useConstructor(mockMDP, mockTrainingListeners, 0, 0)
                .defaultAnswer(Mockito.CALLS_REAL_METHODS));

        int maxEpochStep = 100;
        int nstep = 5;

        when(mockAsyncGlobal.isTrainingComplete()).thenReturn(false);
        when(thread.getAsyncGlobal()).thenReturn(mockAsyncGlobal);

        when(mockAsyncConfiguration.getMaxEpochStep()).thenReturn(maxEpochStep);
        when(mockAsyncConfiguration.getNstep()).thenReturn(nstep);
        when(thread.getConf()).thenReturn(mockAsyncConfiguration);

        when(thread.trainSubEpoch(any(Observation.class), eq(nstep)))
                .thenReturn(new AsyncThread.SubEpochReturn(nstep, mockObs, 0.0, 0.0, true));

        // Act
        thread.run();

        // Assert
        verify(mockNeuralNet, times(1)).reset();
    }

//    @Test
//    public void when_onNewEpochReturnsStop_expect_threadStopped() {
//        // Arrange
//        int stopAfterNumCalls = 1;
//        TestContext context = new TestContext(100000);
//        context.listener.setRemainingOnNewEpochCallCount(stopAfterNumCalls);
//
//        // Act
//        context.sut.run();
//
//        // Assert
//        assertEquals(stopAfterNumCalls + 1, context.listener.onNewEpochCallCount); // +1: The call that returns stop is counted
//        assertEquals(stopAfterNumCalls, context.listener.onEpochTrainingResultCallCount);
//    }
//
//    @Test
//    public void when_epochTrainingResultReturnsStop_expect_threadStopped() {
//        // Arrange
//        int stopAfterNumCalls = 1;
//        TestContext context = new TestContext(100000);
//        context.listener.setRemainingOnEpochTrainingResult(stopAfterNumCalls);
//
//        // Act
//        context.sut.run();
//
//        // Assert
//        assertEquals(stopAfterNumCalls + 1, context.listener.onEpochTrainingResultCallCount); // +1: The call that returns stop is counted
//        assertEquals(stopAfterNumCalls + 1, context.listener.onNewEpochCallCount); // +1: onNewEpoch is called on the epoch that onEpochTrainingResult() will stop
//    }
//
//    @Test
//    public void when_run_expect_preAndPostEpochCalled() {
//        // Arrange
//        int numberOfEpochs = 5;
//        TestContext context = new TestContext(numberOfEpochs);
//
//        // Act
//        context.sut.run();
//
//        // Assert
//        assertEquals(numberOfEpochs, context.sut.preEpochCallCount);
//        assertEquals(numberOfEpochs, context.sut.postEpochCallCount);
//    }
//
//    @Test
//    public void when_run_expect_trainSubEpochCalledAndResultPassedToListeners() {
//        // Arrange
//        int numberOfEpochs = 5;
//        TestContext context = new TestContext(numberOfEpochs);
//
//        // Act
//        context.sut.run();
//
//        // Assert
//        assertEquals(numberOfEpochs, context.listener.statEntries.size());
//        int[] expectedStepCounter = new int[]{10, 20, 30, 40, 50};
//        double expectedReward = (1.0 + 2.0 + 3.0 + 4.0 + 5.0 + 6.0 + 7.0 + 8.0) // reward from init
//                + 1.0; // Reward from trainSubEpoch()
//        for (int i = 0; i < numberOfEpochs; ++i) {
//            IDataManager.StatEntry statEntry = context.listener.statEntries.get(i);
//            assertEquals(expectedStepCounter[i], statEntry.getStepCounter());
//            assertEquals(i, statEntry.getEpochCounter());
//            assertEquals(expectedReward, statEntry.getReward(), 0.0001);
//        }
//    }
//
//    @Test
//    public void when_run_expect_trainSubEpochCalled() {
//        // Arrange
//        int numberOfEpochs = 5;
//        TestContext context = new TestContext(numberOfEpochs);
//
//        // Act
//        context.sut.run();
//
//        // Assert
//        assertEquals(numberOfEpochs, context.sut.trainSubEpochParams.size());
//        double[] expectedObservation = new double[]{0.0, 2.0, 4.0, 6.0, 8.0};
//        for (int i = 0; i < context.sut.trainSubEpochParams.size(); ++i) {
//            MockAsyncThread.TrainSubEpochParams params = context.sut.trainSubEpochParams.get(i);
//            assertEquals(2, params.nstep);
//            assertEquals(expectedObservation.length, params.obs.getData().shape()[1]);
//            for (int j = 0; j < expectedObservation.length; ++j) {
//                assertEquals(expectedObservation[j], 255.0 * params.obs.getData().getDouble(j), 0.00001);
//            }
//        }
//    }

}
