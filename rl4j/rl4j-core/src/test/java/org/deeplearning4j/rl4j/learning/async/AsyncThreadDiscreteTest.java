/*******************************************************************************
 * Copyright (c) 2020 Konduit K. K.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

package org.deeplearning4j.rl4j.learning.async;

import org.deeplearning4j.gym.StepReply;
import org.deeplearning4j.rl4j.learning.IHistoryProcessor;
import org.deeplearning4j.rl4j.learning.configuration.IAsyncLearningConfiguration;
import org.deeplearning4j.rl4j.learning.listener.TrainingListenerList;
import org.deeplearning4j.rl4j.mdp.MDP;
import org.deeplearning4j.rl4j.network.NeuralNet;
import org.deeplearning4j.rl4j.observation.Observation;
import org.deeplearning4j.rl4j.policy.Policy;
import org.deeplearning4j.rl4j.space.Box;
import org.deeplearning4j.rl4j.space.DiscreteSpace;
import org.deeplearning4j.rl4j.space.Encodable;
import org.deeplearning4j.rl4j.space.ObservationSpace;
import org.deeplearning4j.rl4j.util.LegacyMDPWrapper;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.mockito.Mock;
import org.mockito.Mockito;
import org.mockito.junit.MockitoJUnitRunner;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

@RunWith(MockitoJUnitRunner.class)
public class AsyncThreadDiscreteTest {


    AsyncThreadDiscrete<Encodable, NeuralNet> asyncThreadDiscrete;

    @Mock
    IAsyncLearningConfiguration mockAsyncConfiguration;

    @Mock
    UpdateAlgorithm<NeuralNet> mockUpdateAlgorithm;

    @Mock
    IAsyncGlobal<NeuralNet> mockAsyncGlobal;

    @Mock
    NeuralNet mockGlobalCurrentNetwork;

    @Mock
    Policy<Encodable, Integer> mockGlobalCurrentPolicy;

    @Mock
    NeuralNet mockGlobalTargetNetwork;

    @Mock
    MDP<Encodable, Integer, DiscreteSpace> mockMDP;

    @Mock
    LegacyMDPWrapper<Encodable, Integer, DiscreteSpace> mockLegacyMDPWrapper;

    @Mock
    DiscreteSpace mockActionSpace;

    @Mock
    ObservationSpace<Encodable> mockObservationSpace;

    @Mock
    TrainingListenerList mockTrainingListenerList;

    @Mock
    Observation mockObservation;

    int[] observationShape = new int[]{3, 10, 10};
    int actionSize = 4;

    private void setupMDPMocks() {

        when(mockObservationSpace.getShape()).thenReturn(observationShape);
        when(mockActionSpace.noOp()).thenReturn(0);

        when(mockMDP.getObservationSpace()).thenReturn(mockObservationSpace);
        when(mockMDP.getActionSpace()).thenReturn(mockActionSpace);

        int dataLength = 1;
        for (int d : observationShape) {
            dataLength *= d;
        }

        //when(mockMDP.reset()).thenReturn(new Box(new double[dataLength]));

    }

    private void setupCurrentAndTargetMocks() {
        when(mockAsyncGlobal.getTarget()).thenReturn(mockGlobalTargetNetwork);

        when(mockGlobalCurrentNetwork.clone()).thenReturn(mockGlobalCurrentNetwork);
    }

    @Before
    public void setup() {

        setupMDPMocks();
        setupCurrentAndTargetMocks();

        asyncThreadDiscrete = mock(AsyncThreadDiscrete.class, Mockito.withSettings()
                .useConstructor(mockAsyncGlobal, mockMDP, mockTrainingListenerList, 0, 0)
                .defaultAnswer(Mockito.CALLS_REAL_METHODS));

        asyncThreadDiscrete.setUpdateAlgorithm(mockUpdateAlgorithm);

        when(asyncThreadDiscrete.getConf()).thenReturn(mockAsyncConfiguration);
        when(mockAsyncConfiguration.getRewardFactor()).thenReturn(1.0);
        when(asyncThreadDiscrete.getAsyncGlobal()).thenReturn(mockAsyncGlobal);
        when(asyncThreadDiscrete.getPolicy(eq(mockGlobalCurrentNetwork))).thenReturn(mockGlobalCurrentPolicy);

        when(mockGlobalCurrentPolicy.nextAction(eq(mockObservation))).thenReturn(0);

        when(asyncThreadDiscrete.getLegacyMDPWrapper()).thenReturn(mockLegacyMDPWrapper);
        when(mockLegacyMDPWrapper.step(0)).thenReturn(new StepReply<>(mockObservation, 0.0, false, null));
    }

    @Test
    public void when_episodeShorterThanNsteps_returnEpisodeLength() {

        // Arrange
        int episodeRemaining = 4;
        int nSteps = 5;

        // return done after 4 steps (the episode finishes before nsteps)
        when(mockMDP.isDone()).thenAnswer(invocation ->
            asyncThreadDiscrete.getStepCount() == episodeRemaining
        );

        // Act
        AsyncThread.SubEpochReturn subEpochReturn = asyncThreadDiscrete.trainSubEpoch(mockObservation, nSteps);

        // Assert
        assertTrue(subEpochReturn.isEpisodeComplete());
        assertEquals(episodeRemaining, subEpochReturn.getSteps());
    }

    @Test
    public void when_episodeLongerThanNsteps_returnNstepLength() {

        // Arrange
        int episodeRemaining = 5;
        int nSteps = 4;

        // return done after 4 steps (the episode finishes before nsteps)
        when(mockMDP.isDone()).thenAnswer(invocation ->
                asyncThreadDiscrete.getStepCount() == episodeRemaining
        );

        // Act
        AsyncThread.SubEpochReturn subEpochReturn = asyncThreadDiscrete.trainSubEpoch(mockObservation, nSteps);

        // Assert
        assertFalse(subEpochReturn.isEpisodeComplete());
        assertEquals(nSteps, subEpochReturn.getSteps());
    }

}
