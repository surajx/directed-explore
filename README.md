## Count-based Exploration in Feature Space for Reinforcement Learning

We introduce a new count-based optimistic exploration algorithm for Reinforcement Learning (RL) that is feasible in environments with high-dimensional state-action spaces. The success of RL algorithms in these domains depends crucially on generalisation from limited training experience. Function approximation techniques enable RL agents to generalise in order to estimate the value of unvisited states, but at present few methods enable generalisation regarding uncertainty. This has prevented the combination of scalable RL algorithms with efficient exploration strategies that drive the agent to reduce its uncertainty. We present a new method for computing a generalised state visit-count, which allows the agent to estimate the uncertainty associated with any state. Our \phi-pseudocount achieves generalisation by exploiting same feature representation of the state space that is used for value function approximation. States that have less frequently observed features are deemed more uncertain. The \phi-Exploration-Bonus algorithm rewards the agent for exploring in feature space rather than in the untransformed state space. The method is simpler and less computationally expensive than some previous proposals, and achieves near state-of-the-art results on high-dimensional RL benchmarks.

[Paper](/documents/IJCAI_paper.pdf)
[Poster](/documents/IJCAI_Poster.pdf)
[Slides](/documents/IJCAI_slides.pdf)

##### Agent Video (YouTube)
[![Atari Gameplay video](http://img.youtube.com/vi/BoaWbdZphSI/0.jpg)](http://www.youtube.com/watch?v=BoaWbdZphSI)
