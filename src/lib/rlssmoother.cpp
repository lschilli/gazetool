#include "rlssmoother.h"

RlsSmoother::RlsSmoother(double windowSize, double forgetting, double cost)
    : rls(windowSize, forgetting, cost)
{

}

RlsSmoother::RlsSmoother() : rls(30, 0.99, 0.000005)
{

}


RlsSmoother::~RlsSmoother()
{

}

void RlsSmoother::smoothValue(boost::optional<double>& value) {
    boost::optional<double> nextVal;
    if (rlsready) {
        nextVal = rls.get_predicted_next_state()(0);
    }
    if (value.is_initialized()) {
        dlib::matrix<double,1,1> rlsUpdVal;
        rlsUpdVal(0) = value.get();
        rls.update(rlsUpdVal);
        rlsready = true;
    } else {
        rls.update();
    }
    value = nextVal;
}
