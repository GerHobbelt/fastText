ARG MANYLINUX_TAG="2022-10-25-fbea779"
FROM quay.io/pypa/manylinux_2_28_x86_64:${MANYLINUX_TAG}

# ARG GCC_VERSION
# ARG QT_VERSION
# ARG QT_VERSION_MAJOR
# ARG QT_VERSION_MINOR
# ARG QT_VERSION_PATCH
# ARG QT_FULL_VERSION
# ARG PYTHON_VERSION
# ARG PYTHON_SHORT_VERSION
# ARG PYTHON_FULL_VERSION
# ENV PATH="/opt/python/${PYTHON_SHORT_VERSION}/bin:${PATH}"

RUN yum install -y wget ninja-build boost-devel.x86_64

ENV BUILD_TARGET=/build
ENV SRC=/src

RUN mkdir -p "$BUILD_TARGET"

WORKDIR ${BUILD_TARGET}
COPY CMakeLists.txt fasttext.pc.in ${SRC}/
COPY src ${SRC}/src/
# COPY . ${SRC}/
RUN ls /src
RUN cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/usr ../src -G Ninja && cmake --build . --parallel && cmake --install .

