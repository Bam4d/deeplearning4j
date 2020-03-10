PLATFORM=linux-x86_64
CUDA_VERSION=10.2
COMPUTE_CAPABILITY=75

mvn install -Dmaven.test.skip -Dlibnd4j.cuda=${CUDA_VERSION} -Dlibnd4j.compute=${COMPUTE_CAPABILITY} -Dlibnd4j.platform=${PLATFORM} -Djavacpp.platform=${PLATFORM} -pl '!libnd4j' -Dmaven.javadoc.skip=true
