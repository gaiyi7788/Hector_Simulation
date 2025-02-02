#include "../../include/FSM/FSMState.h"

FSMState::FSMState(std::shared_ptr<ControlFSMData> data, FSMStateName stateName, std::string stateNameStr):
            _data(data), _stateName(stateName), _stateNameStr(stateNameStr)
{
    _lowCmd = _data->_lowCmd;
    _lowState = _data->_lowState;
}