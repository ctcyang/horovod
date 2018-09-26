// Copyright 2018 Uber Technologies, Inc. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================

// automatically generated by the FlatBuffers compiler, do not modify


#ifndef FLATBUFFERS_GENERATED_MPIMESSAGE_HOROVOD_COMMON_WIRE_H_
#define FLATBUFFERS_GENERATED_MPIMESSAGE_HOROVOD_COMMON_WIRE_H_

#include "flatbuffers/flatbuffers.h"

namespace horovod {
namespace common {
namespace wire {

struct MPIRequest;

struct MPIRequestList;

struct MPIResponse;

struct MPIResponseList;

enum MPIDataType {
  MPIDataType_HOROVOD_UINT8 = 0,
  MPIDataType_HOROVOD_INT8 = 1,
  MPIDataType_HOROVOD_UINT16 = 2,
  MPIDataType_HOROVOD_INT16 = 3,
  MPIDataType_HOROVOD_INT32 = 4,
  MPIDataType_HOROVOD_INT64 = 5,
  MPIDataType_HOROVOD_FLOAT32 = 6,
  MPIDataType_HOROVOD_FLOAT64 = 7,
  MPIDataType_HOROVOD_BOOL = 8,
  MPIDataType_MIN = MPIDataType_HOROVOD_UINT8,
  MPIDataType_MAX = MPIDataType_HOROVOD_BOOL
};

inline const char **EnumNamesMPIDataType() {
  static const char *names[] = {
    "HOROVOD_UINT8",
    "HOROVOD_INT8",
    "HOROVOD_UINT16",
    "HOROVOD_INT16",
    "HOROVOD_INT32",
    "HOROVOD_INT64",
    "HOROVOD_FLOAT32",
    "HOROVOD_FLOAT64",
    "HOROVOD_BOOL",
    nullptr
  };
  return names;
}

inline const char *EnumNameMPIDataType(MPIDataType e) {
  const size_t index = static_cast<int>(e);
  return EnumNamesMPIDataType()[index];
}

enum MPIRequestType {
  MPIRequestType_ALLREDUCE = 0,
  MPIRequestType_ALLGATHER = 1,
  MPIRequestType_BROADCAST = 2,
  MPIRequestType_MIN = MPIRequestType_ALLREDUCE,
  MPIRequestType_MAX = MPIRequestType_BROADCAST
};

inline const char **EnumNamesMPIRequestType() {
  static const char *names[] = {
    "ALLREDUCE",
    "ALLGATHER",
    "BROADCAST",
    nullptr
  };
  return names;
}

inline const char *EnumNameMPIRequestType(MPIRequestType e) {
  const size_t index = static_cast<int>(e);
  return EnumNamesMPIRequestType()[index];
}

enum MPIResponseType {
  MPIResponseType_ALLREDUCE = 0,
  MPIResponseType_ALLGATHER = 1,
  MPIResponseType_BROADCAST = 2,
  MPIResponseType_ERROR = 3,
  MPIResponseType_MIN = MPIResponseType_ALLREDUCE,
  MPIResponseType_MAX = MPIResponseType_ERROR
};

inline const char **EnumNamesMPIResponseType() {
  static const char *names[] = {
    "ALLREDUCE",
    "ALLGATHER",
    "BROADCAST",
    "ERROR",
    nullptr
  };
  return names;
}

inline const char *EnumNameMPIResponseType(MPIResponseType e) {
  const size_t index = static_cast<int>(e);
  return EnumNamesMPIResponseType()[index];
}

struct MPIRequest FLATBUFFERS_FINAL_CLASS : private flatbuffers::Table {
  enum {
    VT_REQUEST_RANK = 4,
    VT_REQUEST_TYPE = 6,
    VT_TENSOR_TYPE = 8,
    VT_TENSOR_NAME = 10,
    VT_ROOT_RANK = 12,
    VT_DEVICE = 14,
    VT_TENSOR_SHAPE = 16
  };
  int32_t request_rank() const {
    return GetField<int32_t>(VT_REQUEST_RANK, 0);
  }
  MPIRequestType request_type() const {
    return static_cast<MPIRequestType>(GetField<int8_t>(VT_REQUEST_TYPE, 0));
  }
  MPIDataType tensor_type() const {
    return static_cast<MPIDataType>(GetField<int8_t>(VT_TENSOR_TYPE, 0));
  }
  const flatbuffers::String *tensor_name() const {
    return GetPointer<const flatbuffers::String *>(VT_TENSOR_NAME);
  }
  int32_t root_rank() const {
    return GetField<int32_t>(VT_ROOT_RANK, 0);
  }
  int32_t device() const {
    return GetField<int32_t>(VT_DEVICE, 0);
  }
  const flatbuffers::Vector<int64_t> *tensor_shape() const {
    return GetPointer<const flatbuffers::Vector<int64_t> *>(VT_TENSOR_SHAPE);
  }
  bool Verify(flatbuffers::Verifier &verifier) const {
    return VerifyTableStart(verifier) &&
           VerifyField<int32_t>(verifier, VT_REQUEST_RANK) &&
           VerifyField<int8_t>(verifier, VT_REQUEST_TYPE) &&
           VerifyField<int8_t>(verifier, VT_TENSOR_TYPE) &&
           VerifyField<flatbuffers::uoffset_t>(verifier, VT_TENSOR_NAME) &&
           verifier.Verify(tensor_name()) &&
           VerifyField<int32_t>(verifier, VT_ROOT_RANK) &&
           VerifyField<int32_t>(verifier, VT_DEVICE) &&
           VerifyField<flatbuffers::uoffset_t>(verifier, VT_TENSOR_SHAPE) &&
           verifier.Verify(tensor_shape()) &&
           verifier.EndTable();
  }
};

struct MPIRequestBuilder {
  flatbuffers::FlatBufferBuilder &fbb_;
  flatbuffers::uoffset_t start_;
  void add_request_rank(int32_t request_rank) {
    fbb_.AddElement<int32_t>(MPIRequest::VT_REQUEST_RANK, request_rank, 0);
  }
  void add_request_type(MPIRequestType request_type) {
    fbb_.AddElement<int8_t>(MPIRequest::VT_REQUEST_TYPE, static_cast<int8_t>(request_type), 0);
  }
  void add_tensor_type(MPIDataType tensor_type) {
    fbb_.AddElement<int8_t>(MPIRequest::VT_TENSOR_TYPE, static_cast<int8_t>(tensor_type), 0);
  }
  void add_tensor_name(flatbuffers::Offset<flatbuffers::String> tensor_name) {
    fbb_.AddOffset(MPIRequest::VT_TENSOR_NAME, tensor_name);
  }
  void add_root_rank(int32_t root_rank) {
    fbb_.AddElement<int32_t>(MPIRequest::VT_ROOT_RANK, root_rank, 0);
  }
  void add_device(int32_t device) {
    fbb_.AddElement<int32_t>(MPIRequest::VT_DEVICE, device, 0);
  }
  void add_tensor_shape(flatbuffers::Offset<flatbuffers::Vector<int64_t>> tensor_shape) {
    fbb_.AddOffset(MPIRequest::VT_TENSOR_SHAPE, tensor_shape);
  }
  MPIRequestBuilder(flatbuffers::FlatBufferBuilder &_fbb)
        : fbb_(_fbb) {
    start_ = fbb_.StartTable();
  }
  MPIRequestBuilder &operator=(const MPIRequestBuilder &);
  flatbuffers::Offset<MPIRequest> Finish() {
    const auto end = fbb_.EndTable(start_, 7);
    auto o = flatbuffers::Offset<MPIRequest>(end);
    return o;
  }
};

inline flatbuffers::Offset<MPIRequest> CreateMPIRequest(
    flatbuffers::FlatBufferBuilder &_fbb,
    int32_t request_rank = 0,
    MPIRequestType request_type = MPIRequestType_ALLREDUCE,
    MPIDataType tensor_type = MPIDataType_HOROVOD_UINT8,
    flatbuffers::Offset<flatbuffers::String> tensor_name = 0,
    int32_t root_rank = 0,
    int32_t device = 0,
    flatbuffers::Offset<flatbuffers::Vector<int64_t>> tensor_shape = 0) {
  MPIRequestBuilder builder_(_fbb);
  builder_.add_tensor_shape(tensor_shape);
  builder_.add_device(device);
  builder_.add_root_rank(root_rank);
  builder_.add_tensor_name(tensor_name);
  builder_.add_request_rank(request_rank);
  builder_.add_tensor_type(tensor_type);
  builder_.add_request_type(request_type);
  return builder_.Finish();
}

inline flatbuffers::Offset<MPIRequest> CreateMPIRequestDirect(
    flatbuffers::FlatBufferBuilder &_fbb,
    int32_t request_rank = 0,
    MPIRequestType request_type = MPIRequestType_ALLREDUCE,
    MPIDataType tensor_type = MPIDataType_HOROVOD_UINT8,
    const char *tensor_name = nullptr,
    int32_t root_rank = 0,
    int32_t device = 0,
    const std::vector<int64_t> *tensor_shape = nullptr) {
  return horovod::common::wire::CreateMPIRequest(
      _fbb,
      request_rank,
      request_type,
      tensor_type,
      tensor_name ? _fbb.CreateString(tensor_name) : 0,
      root_rank,
      device,
      tensor_shape ? _fbb.CreateVector<int64_t>(*tensor_shape) : 0);
}

struct MPIRequestList FLATBUFFERS_FINAL_CLASS : private flatbuffers::Table {
  enum {
    VT_REQUESTS = 4,
    VT_SHUTDOWN = 6
  };
  const flatbuffers::Vector<flatbuffers::Offset<MPIRequest>> *requests() const {
    return GetPointer<const flatbuffers::Vector<flatbuffers::Offset<MPIRequest>> *>(VT_REQUESTS);
  }
  bool shutdown() const {
    return GetField<uint8_t>(VT_SHUTDOWN, 0) != 0;
  }
  bool Verify(flatbuffers::Verifier &verifier) const {
    return VerifyTableStart(verifier) &&
           VerifyField<flatbuffers::uoffset_t>(verifier, VT_REQUESTS) &&
           verifier.Verify(requests()) &&
           verifier.VerifyVectorOfTables(requests()) &&
           VerifyField<uint8_t>(verifier, VT_SHUTDOWN) &&
           verifier.EndTable();
  }
};

struct MPIRequestListBuilder {
  flatbuffers::FlatBufferBuilder &fbb_;
  flatbuffers::uoffset_t start_;
  void add_requests(flatbuffers::Offset<flatbuffers::Vector<flatbuffers::Offset<MPIRequest>>> requests) {
    fbb_.AddOffset(MPIRequestList::VT_REQUESTS, requests);
  }
  void add_shutdown(bool shutdown) {
    fbb_.AddElement<uint8_t>(MPIRequestList::VT_SHUTDOWN, static_cast<uint8_t>(shutdown), 0);
  }
  MPIRequestListBuilder(flatbuffers::FlatBufferBuilder &_fbb)
        : fbb_(_fbb) {
    start_ = fbb_.StartTable();
  }
  MPIRequestListBuilder &operator=(const MPIRequestListBuilder &);
  flatbuffers::Offset<MPIRequestList> Finish() {
    const auto end = fbb_.EndTable(start_, 2);
    auto o = flatbuffers::Offset<MPIRequestList>(end);
    return o;
  }
};

inline flatbuffers::Offset<MPIRequestList> CreateMPIRequestList(
    flatbuffers::FlatBufferBuilder &_fbb,
    flatbuffers::Offset<flatbuffers::Vector<flatbuffers::Offset<MPIRequest>>> requests = 0,
    bool shutdown = false) {
  MPIRequestListBuilder builder_(_fbb);
  builder_.add_requests(requests);
  builder_.add_shutdown(shutdown);
  return builder_.Finish();
}

inline flatbuffers::Offset<MPIRequestList> CreateMPIRequestListDirect(
    flatbuffers::FlatBufferBuilder &_fbb,
    const std::vector<flatbuffers::Offset<MPIRequest>> *requests = nullptr,
    bool shutdown = false) {
  return horovod::common::wire::CreateMPIRequestList(
      _fbb,
      requests ? _fbb.CreateVector<flatbuffers::Offset<MPIRequest>>(*requests) : 0,
      shutdown);
}

struct MPIResponse FLATBUFFERS_FINAL_CLASS : private flatbuffers::Table {
  enum {
    VT_RESPONSE_TYPE = 4,
    VT_TENSOR_NAMES = 6,
    VT_ERROR_MESSAGE = 8,
    VT_DEVICES = 10,
    VT_TENSOR_SIZES = 12
  };
  MPIResponseType response_type() const {
    return static_cast<MPIResponseType>(GetField<int8_t>(VT_RESPONSE_TYPE, 0));
  }
  const flatbuffers::Vector<flatbuffers::Offset<flatbuffers::String>> *tensor_names() const {
    return GetPointer<const flatbuffers::Vector<flatbuffers::Offset<flatbuffers::String>> *>(VT_TENSOR_NAMES);
  }
  const flatbuffers::String *error_message() const {
    return GetPointer<const flatbuffers::String *>(VT_ERROR_MESSAGE);
  }
  const flatbuffers::Vector<int32_t> *devices() const {
    return GetPointer<const flatbuffers::Vector<int32_t> *>(VT_DEVICES);
  }
  const flatbuffers::Vector<int64_t> *tensor_sizes() const {
    return GetPointer<const flatbuffers::Vector<int64_t> *>(VT_TENSOR_SIZES);
  }
  bool Verify(flatbuffers::Verifier &verifier) const {
    return VerifyTableStart(verifier) &&
           VerifyField<int8_t>(verifier, VT_RESPONSE_TYPE) &&
           VerifyField<flatbuffers::uoffset_t>(verifier, VT_TENSOR_NAMES) &&
           verifier.Verify(tensor_names()) &&
           verifier.VerifyVectorOfStrings(tensor_names()) &&
           VerifyField<flatbuffers::uoffset_t>(verifier, VT_ERROR_MESSAGE) &&
           verifier.Verify(error_message()) &&
           VerifyField<flatbuffers::uoffset_t>(verifier, VT_DEVICES) &&
           verifier.Verify(devices()) &&
           VerifyField<flatbuffers::uoffset_t>(verifier, VT_TENSOR_SIZES) &&
           verifier.Verify(tensor_sizes()) &&
           verifier.EndTable();
  }
};

struct MPIResponseBuilder {
  flatbuffers::FlatBufferBuilder &fbb_;
  flatbuffers::uoffset_t start_;
  void add_response_type(MPIResponseType response_type) {
    fbb_.AddElement<int8_t>(MPIResponse::VT_RESPONSE_TYPE, static_cast<int8_t>(response_type), 0);
  }
  void add_tensor_names(flatbuffers::Offset<flatbuffers::Vector<flatbuffers::Offset<flatbuffers::String>>> tensor_names) {
    fbb_.AddOffset(MPIResponse::VT_TENSOR_NAMES, tensor_names);
  }
  void add_error_message(flatbuffers::Offset<flatbuffers::String> error_message) {
    fbb_.AddOffset(MPIResponse::VT_ERROR_MESSAGE, error_message);
  }
  void add_devices(flatbuffers::Offset<flatbuffers::Vector<int32_t>> devices) {
    fbb_.AddOffset(MPIResponse::VT_DEVICES, devices);
  }
  void add_tensor_sizes(flatbuffers::Offset<flatbuffers::Vector<int64_t>> tensor_sizes) {
    fbb_.AddOffset(MPIResponse::VT_TENSOR_SIZES, tensor_sizes);
  }
  MPIResponseBuilder(flatbuffers::FlatBufferBuilder &_fbb)
        : fbb_(_fbb) {
    start_ = fbb_.StartTable();
  }
  MPIResponseBuilder &operator=(const MPIResponseBuilder &);
  flatbuffers::Offset<MPIResponse> Finish() {
    const auto end = fbb_.EndTable(start_, 5);
    auto o = flatbuffers::Offset<MPIResponse>(end);
    return o;
  }
};

inline flatbuffers::Offset<MPIResponse> CreateMPIResponse(
    flatbuffers::FlatBufferBuilder &_fbb,
    MPIResponseType response_type = MPIResponseType_ALLREDUCE,
    flatbuffers::Offset<flatbuffers::Vector<flatbuffers::Offset<flatbuffers::String>>> tensor_names = 0,
    flatbuffers::Offset<flatbuffers::String> error_message = 0,
    flatbuffers::Offset<flatbuffers::Vector<int32_t>> devices = 0,
    flatbuffers::Offset<flatbuffers::Vector<int64_t>> tensor_sizes = 0) {
  MPIResponseBuilder builder_(_fbb);
  builder_.add_tensor_sizes(tensor_sizes);
  builder_.add_devices(devices);
  builder_.add_error_message(error_message);
  builder_.add_tensor_names(tensor_names);
  builder_.add_response_type(response_type);
  return builder_.Finish();
}

inline flatbuffers::Offset<MPIResponse> CreateMPIResponseDirect(
    flatbuffers::FlatBufferBuilder &_fbb,
    MPIResponseType response_type = MPIResponseType_ALLREDUCE,
    const std::vector<flatbuffers::Offset<flatbuffers::String>> *tensor_names = nullptr,
    const char *error_message = nullptr,
    const std::vector<int32_t> *devices = nullptr,
    const std::vector<int64_t> *tensor_sizes = nullptr) {
  return horovod::common::wire::CreateMPIResponse(
      _fbb,
      response_type,
      tensor_names ? _fbb.CreateVector<flatbuffers::Offset<flatbuffers::String>>(*tensor_names) : 0,
      error_message ? _fbb.CreateString(error_message) : 0,
      devices ? _fbb.CreateVector<int32_t>(*devices) : 0,
      tensor_sizes ? _fbb.CreateVector<int64_t>(*tensor_sizes) : 0);
}

struct MPIResponseList FLATBUFFERS_FINAL_CLASS : private flatbuffers::Table {
  enum {
    VT_RESPONSES = 4,
    VT_SHUTDOWN = 6
  };
  const flatbuffers::Vector<flatbuffers::Offset<MPIResponse>> *responses() const {
    return GetPointer<const flatbuffers::Vector<flatbuffers::Offset<MPIResponse>> *>(VT_RESPONSES);
  }
  bool shutdown() const {
    return GetField<uint8_t>(VT_SHUTDOWN, 0) != 0;
  }
  bool Verify(flatbuffers::Verifier &verifier) const {
    return VerifyTableStart(verifier) &&
           VerifyField<flatbuffers::uoffset_t>(verifier, VT_RESPONSES) &&
           verifier.Verify(responses()) &&
           verifier.VerifyVectorOfTables(responses()) &&
           VerifyField<uint8_t>(verifier, VT_SHUTDOWN) &&
           verifier.EndTable();
  }
};

struct MPIResponseListBuilder {
  flatbuffers::FlatBufferBuilder &fbb_;
  flatbuffers::uoffset_t start_;
  void add_responses(flatbuffers::Offset<flatbuffers::Vector<flatbuffers::Offset<MPIResponse>>> responses) {
    fbb_.AddOffset(MPIResponseList::VT_RESPONSES, responses);
  }
  void add_shutdown(bool shutdown) {
    fbb_.AddElement<uint8_t>(MPIResponseList::VT_SHUTDOWN, static_cast<uint8_t>(shutdown), 0);
  }
  MPIResponseListBuilder(flatbuffers::FlatBufferBuilder &_fbb)
        : fbb_(_fbb) {
    start_ = fbb_.StartTable();
  }
  MPIResponseListBuilder &operator=(const MPIResponseListBuilder &);
  flatbuffers::Offset<MPIResponseList> Finish() {
    const auto end = fbb_.EndTable(start_, 2);
    auto o = flatbuffers::Offset<MPIResponseList>(end);
    return o;
  }
};

inline flatbuffers::Offset<MPIResponseList> CreateMPIResponseList(
    flatbuffers::FlatBufferBuilder &_fbb,
    flatbuffers::Offset<flatbuffers::Vector<flatbuffers::Offset<MPIResponse>>> responses = 0,
    bool shutdown = false) {
  MPIResponseListBuilder builder_(_fbb);
  builder_.add_responses(responses);
  builder_.add_shutdown(shutdown);
  return builder_.Finish();
}

inline flatbuffers::Offset<MPIResponseList> CreateMPIResponseListDirect(
    flatbuffers::FlatBufferBuilder &_fbb,
    const std::vector<flatbuffers::Offset<MPIResponse>> *responses = nullptr,
    bool shutdown = false) {
  return horovod::common::wire::CreateMPIResponseList(
      _fbb,
      responses ? _fbb.CreateVector<flatbuffers::Offset<MPIResponse>>(*responses) : 0,
      shutdown);
}

}  // namespace wire
}  // namespace common
}  // namespace horovod

#endif  // FLATBUFFERS_GENERATED_MPIMESSAGE_HOROVOD_COMMON_WIRE_H_
