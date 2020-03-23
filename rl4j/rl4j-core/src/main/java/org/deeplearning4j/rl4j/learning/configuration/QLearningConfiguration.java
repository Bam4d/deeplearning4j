/*******************************************************************************
 * Copyright (c) 2015-2020 Skymind, Inc.
 *
 *  This program and the accompanying materials are made available under the
 *  terms of the Apache License, Version 2.0 which is available at
 *  https://www.apache.org/licenses/LICENSE-2.0.
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  License for the specific language governing permissions and limitations
 *  under the License.
 *
 *  SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

package org.deeplearning4j.rl4j.learning.configuration;

import lombok.Builder;
import lombok.Data;
import lombok.EqualsAndHashCode;
import lombok.NoArgsConstructor;
import lombok.experimental.SuperBuilder;

@Data
@SuperBuilder
@NoArgsConstructor
@EqualsAndHashCode(callSuper = false)
public class QLearningConfiguration extends LearningConfiguration {

    @Builder.Default
    int expRepMaxSize = 150000;

    @Builder.Default
    int batchSize = 32;

    @Builder.Default
    int targetDqnUpdateFreq = 100;

    @Builder.Default
    int updateStart = 10;

    @Builder.Default
    double errorClamp = 1.0;

    @Builder.Default
    double minEpsilon = 0.1f;

    @Builder.Default
    int epsilonNbStep = 10000;

    @Builder.Default
    boolean doubleDQN = false;

}
