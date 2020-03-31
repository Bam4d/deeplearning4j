package org.deeplearning4j.rl4j.learning.async;

import org.deeplearning4j.rl4j.learning.listener.TrainingListener;
import org.deeplearning4j.rl4j.network.NeuralNet;
import org.deeplearning4j.rl4j.space.ActionSpace;
import org.deeplearning4j.rl4j.space.Box;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.mockito.Mock;
import org.mockito.Mockito;
import org.mockito.junit.MockitoJUnitRunner;
import org.nd4j.linalg.api.ndarray.INDArray;

import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;


@RunWith(MockitoJUnitRunner.class)
public class AsyncLearningTest {

    AsyncLearning<Box, INDArray, ActionSpace<INDArray>, NeuralNet> asyncLearning;

    @Mock
    TrainingListener mockTrainingListener;

    @Mock
    AsyncGlobal<NeuralNet> mockAsyncGlobal;

    @Before
    public void setup() {
        asyncLearning = mock(AsyncLearning.class, Mockito.withSettings()
                .defaultAnswer(Mockito.CALLS_REAL_METHODS));

        asyncLearning.addListener(mockTrainingListener);

        when(asyncLearning.getAsyncGlobal()).thenReturn(mockAsyncGlobal);
    }

    @Test
    public void when_trainStartReturnsStop_expect_noTraining() {
        // Arrange
        when(mockTrainingListener.onTrainingStart()).thenReturn(TrainingListener.ListenerResponse.STOP);

        // Act
        asyncLearning.train();

        // Assert
        verify(mockTrainingListener, times(1)).onTrainingStart();
        verify(mockTrainingListener, times(1)).onTrainingEnd();
    }

    @Test
    public void when_trainingIsComplete_expect_trainingStop() {
        // Arrange
        when(mockAsyncGlobal.isTrainingComplete()).thenReturn(true);

        // Act
        asyncLearning.train();

        // Assert
        verify(mockTrainingListener, times(1)).onTrainingStart();
        verify(mockTrainingListener, times(1)).onTrainingEnd();
    }

    @Test
    public void when_training_expect_onTrainingProgressCalled() {
        // Arrange
        asyncLearning.setProgressMonitorFrequency(100);

        // Act
        asyncLearning.train();

        // Assert
        verify(mockTrainingListener, times(1)).onTrainingStart();
        verify(mockTrainingListener, times(1)).onTrainingEnd();
        verify(mockTrainingListener, times(1)).onTrainingProgress(eq(asyncLearning));
    }
}
