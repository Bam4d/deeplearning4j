package org.deeplearning4j.rl4j.learning.sync;

import org.deeplearning4j.rl4j.learning.listener.TrainingListener;
import org.deeplearning4j.rl4j.learning.sync.qlearning.QLearning;
import org.deeplearning4j.rl4j.learning.sync.support.MockStatEntry;
import org.deeplearning4j.rl4j.network.NeuralNet;
import org.deeplearning4j.rl4j.space.ActionSpace;
import org.deeplearning4j.rl4j.space.Box;
import org.deeplearning4j.rl4j.util.IDataManager;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.mockito.Mock;
import org.mockito.Mockito;
import org.mockito.junit.MockitoJUnitRunner;
import org.nd4j.linalg.api.ndarray.INDArray;

import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

@RunWith(MockitoJUnitRunner.class)
public class SyncLearningTest {

    @Mock
    TrainingListener mockTrainingListener;

    SyncLearning<Box, INDArray, ActionSpace<INDArray>, NeuralNet> syncLearning;

    @Before
    public void setup() {

        QLearning.QLConfiguration lconfig = QLearning.QLConfiguration.builder().maxStep(10).build();
        syncLearning = mock(SyncLearning.class, Mockito.withSettings()
                .useConstructor(lconfig)
                .defaultAnswer(Mockito.CALLS_REAL_METHODS));

        syncLearning.addListener(mockTrainingListener);

        when(syncLearning.trainEpoch()).thenAnswer(invocation -> {
            syncLearning.incrementEpoch();
            syncLearning.incrementStep();
            return new MockStatEntry(syncLearning.getEpochCount(), syncLearning.getStepCount(), 1.0);
        });
    }

    @Test
    public void when_training_expect_listenersToBeCalled() {

        // Act
        syncLearning.train();

        verify(mockTrainingListener, times(1)).onTrainingStart();
        verify(mockTrainingListener, times(10)).onNewEpoch(eq(syncLearning));
        verify(mockTrainingListener, times(10)).onEpochTrainingResult(eq(syncLearning), any(IDataManager.StatEntry.class));
        verify(mockTrainingListener, times(1)).onTrainingEnd();

    }

    @Test
    public void when_trainingStartCanContinueFalse_expect_trainingStopped() {
        // Arrange
        when(mockTrainingListener.onTrainingStart()).thenReturn(TrainingListener.ListenerResponse.STOP);

        // Act
        syncLearning.train();

        verify(mockTrainingListener, times(1)).onTrainingStart();
        verify(mockTrainingListener, times(0)).onNewEpoch(eq(syncLearning));
        verify(mockTrainingListener, times(0)).onEpochTrainingResult(eq(syncLearning), any(IDataManager.StatEntry.class));
        verify(mockTrainingListener, times(1)).onTrainingEnd();
    }

    @Test
    public void when_newEpochCanContinueFalse_expect_trainingStopped() {
        // Arrange
        when(mockTrainingListener.onNewEpoch(eq(syncLearning)))
                .thenReturn(TrainingListener.ListenerResponse.CONTINUE)
                .thenReturn(TrainingListener.ListenerResponse.CONTINUE)
                .thenReturn(TrainingListener.ListenerResponse.STOP);

        // Act
        syncLearning.train();

        verify(mockTrainingListener, times(1)).onTrainingStart();
        verify(mockTrainingListener, times(3)).onNewEpoch(eq(syncLearning));
        verify(mockTrainingListener, times(2)).onEpochTrainingResult(eq(syncLearning), any(IDataManager.StatEntry.class));
        verify(mockTrainingListener, times(1)).onTrainingEnd();

    }

    @Test
    public void when_epochTrainingResultCanContinueFalse_expect_trainingStopped() {
        // Arrange
        when(mockTrainingListener.onEpochTrainingResult(eq(syncLearning), any(IDataManager.StatEntry.class)))
                .thenReturn(TrainingListener.ListenerResponse.CONTINUE)
                .thenReturn(TrainingListener.ListenerResponse.CONTINUE)
                .thenReturn(TrainingListener.ListenerResponse.STOP);

        // Act
        syncLearning.train();

        verify(mockTrainingListener, times(1)).onTrainingStart();
        verify(mockTrainingListener, times(3)).onNewEpoch(eq(syncLearning));
        verify(mockTrainingListener, times(3)).onEpochTrainingResult(eq(syncLearning), any(IDataManager.StatEntry.class));
        verify(mockTrainingListener, times(1)).onTrainingEnd();
    }
}
