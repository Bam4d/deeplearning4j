/*******************************************************************************
 * Copyright (c) 2015-2019 Skymind, Inc.
 * Copyright (c) 2020 Konduit K.K.
 *
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

import lombok.AccessLevel;
import lombok.Getter;
import lombok.Setter;
import org.deeplearning4j.gym.StepReply;
import org.deeplearning4j.rl4j.experience.ExperienceHandler;
import org.deeplearning4j.rl4j.experience.StateActionExperienceHandler;
import org.deeplearning4j.rl4j.experience.ExperienceHandler;
import org.deeplearning4j.rl4j.experience.StateActionExperienceHandler;
import org.deeplearning4j.rl4j.experience.StateActionPair;
import org.deeplearning4j.rl4j.learning.IHistoryProcessor;
import org.deeplearning4j.rl4j.learning.listener.TrainingListenerList;
import org.deeplearning4j.rl4j.mdp.MDP;
import org.deeplearning4j.rl4j.network.NeuralNet;
import org.deeplearning4j.rl4j.observation.Observation;
import org.deeplearning4j.rl4j.policy.IPolicy;
import org.deeplearning4j.rl4j.space.DiscreteSpace;
import org.deeplearning4j.rl4j.space.Encodable;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Stack;

/**
 * @author rubenfiszel (ruben.fiszel@epfl.ch) on 8/5/16.
 *
 * Async Learning specialized for the Discrete Domain
 *
 */
public abstract class AsyncThreadDiscrete<O extends Encodable, NN extends NeuralNet>
                extends AsyncThread<O, Integer, DiscreteSpace, NN> {

    @Getter
    private NN current;

    @Setter(AccessLevel.PROTECTED)
    private UpdateAlgorithm<NN> updateAlgorithm;

    // TODO: Make it configurable with a builder
    @Setter(AccessLevel.PROTECTED)
    private ExperienceHandler experienceHandler = new StateActionExperienceHandler();

    public AsyncThreadDiscrete(IAsyncGlobal<NN> asyncGlobal,
                               MDP<O, Integer, DiscreteSpace> mdp,
                               TrainingListenerList listeners,
                               int threadNumber,
                               int deviceNum) {
        super(mdp, listeners, threadNumber, deviceNum);
        synchronized (asyncGlobal) {
            current = (NN)asyncGlobal.getCurrent().clone();
        }
    }

    // TODO: Add an actor-learner class and be able to inject the update algorithm
    protected abstract UpdateAlgorithm<NN> buildUpdateAlgorithm();

    @Override
    public void setHistoryProcessor(IHistoryProcessor historyProcessor) {
        super.setHistoryProcessor(historyProcessor);
        updateAlgorithm = buildUpdateAlgorithm();
    }

    @Override
    protected void preEpisode() {
        experienceHandler.reset();
    }


    /**
     * "Subepoch"  correspond to the t_max-step iterations
     * that stack rewards with t_max MiniTrans
     *
     * @param sObs the obs to start from
     * @param nstep the number of max nstep (step until t_max or state is terminal)
     * @return subepoch training informations
     */
    public SubEpochReturn trainSubEpoch(Observation sObs, int nstep) {

        current.copy(getAsyncGlobal().getTarget());

        Observation obs = sObs;
        IPolicy<O, Integer> policy = getPolicy(current);

        Integer action = getMdp().getActionSpace().noOp();
        IHistoryProcessor hp = getHistoryProcessor();
        int skipFrame = hp != null ? hp.getConf().getSkipFrame() : 1;

        double reward = 0;
        double accuReward = 0;

        while (!getMdp().isDone() && !hasCollectedNSteps(experienceHandler, nstep, skipFrame)) {

            //if step of training, just repeat lastAction
            if (!obs.isSkipped()) {
                action = policy.nextAction(obs);
            }

            StepReply<Observation> stepReply = getLegacyMDPWrapper().step(action);
            accuReward += stepReply.getReward() * getConf().getRewardFactor();

            if (!obs.isSkipped()) {
                experienceHandler.addExperience(obs, action, accuReward, stepReply.isDone());
                accuReward = 0;
            }

            obs = stepReply.getObservation();
            reward += stepReply.getReward();

            incrementSteps();

        }

        boolean episodeComplete = getMdp().isDone();

        if (episodeComplete && hasCollectedNSteps(experienceHandler, nstep, skipFrame)) {
            experienceHandler.setFinalObservation(obs);
        }

        int experienceSize = experienceHandler.getTrainingBatchSize();

        getAsyncGlobal().applyGradient(updateAlgorithm.computeGradients(current, experienceHandler.generateTrainingBatch()), experienceSize);

        experienceHandler.reset();

        return new SubEpochReturn(experienceSize, obs, reward, current.getLatestScore(), episodeComplete);
    }

    private boolean hasCollectedNSteps(ExperienceHandler experienceHandler, int nSteps, int skipFrames) {
        int experienceSize = experienceHandler.getExperience().size();
        int updateFrequency = nSteps * skipFrames;
        return experienceSize > 0 && experienceSize % updateFrequency == 0;
    }
}
