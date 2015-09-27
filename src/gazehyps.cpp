#include "gazehyps.h"

GazeHypList::GazeHypList()
{
}

void GazeHypList::waitready()
{
    std::unique_lock<std::mutex> lock(_mutex);
    while(_tasks) {
        _cond.wait(lock);
    }
}

void GazeHypList::setready(int ready)
{
    std::lock_guard<std::mutex> lock(_mutex);
    _tasks += ready;
    if (!_tasks) {
        _cond.notify_one();
    }
}

void GazeHypList::addGazeHyp(GazeHyp &hyp)
{
    _hyps.push_back(hyp);
}

std::vector<GazeHyp>::iterator GazeHypList::begin()
{
    return _hyps.begin();
}

std::vector<GazeHyp>::iterator GazeHypList::end()
{
    return _hyps.end();
}

std::vector<GazeHyp>::const_iterator GazeHypList::begin() const
{
    return _hyps.begin();
}

std::vector<GazeHyp>::const_iterator GazeHypList::end() const
{
    return _hyps.end();
}

size_t GazeHypList::size()
{
    return _hyps.size();
}

GazeHyp &GazeHypList::hyps(int i)
{
    return _hyps[i];
}
