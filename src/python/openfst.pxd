#!python
#cython: language_level=3

""" Openfst python interface.
"""

from libcpp.vector cimport vector
from libcpp.string cimport string
from libcpp.pair cimport pair
from libc.stdint cimport uint64_t, uint32_t
from util cimport ostream, istream

cimport sym

cdef extern from "<fst/fstlib.h>" namespace "fst":
    enum:
        kIDeterministic
        kODeterministic
        kAcceptor
        kTopSorted
        kWeighted

    cdef cppclass Weight:
        pass

    cdef cppclass Arc[W]:
        int ilabel
        int olabel
        int nextstate
        Arc(int ilabel, int olabel, W& weight, int nextstate)
        W weight

    cdef cppclass ArcIterator[T]:
        ArcIterator(T& fst, int state)
        bint Done()
        void Next()
        Arc& Value()

    cdef cppclass Fst:
        int Start()
        unsigned NumArcs(int s)
        Fst* Copy()
        bint Write(string& filename)
        uint64_t Properties(uint64_t mask, bint compute)

    cdef cppclass ExpandedFst(Fst):
        int NumStates()

    cdef cppclass MutableFst(ExpandedFst):
        int AddState()
        void SetFinal(int s, Weight w)
        void SetStart(int s)
        void SetInputSymbols(sym.SymbolTable* isyms)
        void SetOutputSymbols(sym.SymbolTable* osyms)
        sym.SymbolTable* MutableInputSymbols()
        sym.SymbolTable* MutableOutputSymbols()

    cdef cppclass FstHeader:
        void Read(istream& stream, string& source)
        string ArcType()
        string FstType()

    cdef cppclass TropicalWeight(Weight):
        float Value()
        TropicalWeight(float value)
        TropicalWeight(TropicalWeight weight)
        bint operator==(TropicalWeight& other)
        TropicalWeight& set_value "operator=" (TropicalWeight& other)

    cdef TropicalWeight Plus(TropicalWeight &w1, TropicalWeight& w2)
    cdef TropicalWeight Times(TropicalWeight &w1, TropicalWeight& w2)
    cdef TropicalWeight Divide(TropicalWeight &w1, TropicalWeight& w2)

    cdef TropicalWeight TropicalWeightZero "fst::TropicalWeight::Zero" ()
    cdef TropicalWeight TropicalWeightOne "fst::TropicalWeight::One" ()

    cdef bint ApproxEqual(TropicalWeight &w1, TropicalWeight &w2)

    ctypedef Arc[TropicalWeight] StdArc

    cdef cppclass StdVectorFst(MutableFst):
        TropicalWeight Final(int s)
        void AddArc(int s, StdArc &arc)

	# cdef cppclass LogVectorFst "fst::VectorFst<fst::LogArc>" (MutableFst):
	#     LogWeight Final(int s)
	#     void AddArc(int s, LogArc &arc)

    cdef StdVectorFst* StdVectorFstRead "fst::StdVectorFst::Read" (string& filename)
    # cdef LogVectorFst* LogVectorFstRead "fst::VectorFst<fst::LogArc>::Read" (string& filename)

    cdef cppclass ILabelCompare[A]:
        pass

    cdef cppclass OLabelCompare[A]:
        pass

    cdef cppclass ArcMapper:
        pass

    cdef cppclass PlusStdArcMapper "fst::PlusMapper<fst::StdArc>"(ArcMapper):
        PlusStdArcMapper(TropicalWeight)
    cdef cppclass TimesStdArcMapper "fst::TimesMapper<fst::StdArc>"(ArcMapper):
        TimesStdArcMapper(TropicalWeight)
    cdef cppclass InvertTropicalWeightMapper "fst::InvertWeightMapper<fst::StdArc>"(ArcMapper):
        InvertTropicalWeightMapper()
    cdef cppclass RmTropicalWeightMapper "fst::RmWeightMapper<fst::StdArc>"(ArcMapper):
        RmTropicalWeightMapper()
    cdef cppclass LogToStdWeightConvertMapper "fst::WeightConvertMapper<fst::LogArc, fst::StdArc>"(ArcMapper):
        LogToStdWeightConvertMapper()
    cdef cppclass LogProbStdArcSelector "fst::LogProbArcSelector<fst::StdArc>":
        LogProbStdArcSelector(int seed)
    cdef cppclass UniformStdArcSelector "fst::UniformArcSelector<fst::StdArc>":
        UniformStdArcSelector(int seed)
    cdef cppclass RandGenOptions:
        pass
    cdef cppclass LogProbStdArcRandGenOptions "fst::RandGenOptions< fst::LogProbArcSelector<fst::StdArc> >"(RandGenOptions):
        LogProbStdArcRandGenOptions(LogProbStdArcSelector& selector, int maxlen, int npath, bint weighted)
    cdef cppclass UniformStdArcRandGenOptions "fst::RandGenOptions< fst::UniformArcSelector<fst::StdArc> >"(RandGenOptions):
        UniformStdArcRandGenOptions(UniformStdArcSelector& selector, int maxlen, int npath, bint weighted)


    enum ProjectType:
        PROJECT_INPUT
        PROJECT_OUTPUT

    enum ClosureType:
        CLOSURE_STAR
        CLOSURE_PLUS

    enum ReweightType:
        REWEIGHT_TO_INITIAL
        REWEIGHT_TO_FINAL

    enum:
        kPushWeights
        kPushLabels

    cdef bint Equivalent(Fst& fst1, Fst& fst2)

    # Constructive operations
    cdef void Compose(Fst &ifst1, Fst &ifst2, MutableFst* ofst)
    cdef void Determinize(Fst& ifst, MutableFst* ofst)
    cdef void Difference(Fst &ifst1, Fst &ifst2, MutableFst* ofst)
    cdef void Intersect(Fst &ifst1, Fst &ifst2, MutableFst* ofst)
    cdef void Reverse(Fst &ifst, MutableFst* ofst)
    cdef void ShortestPath(Fst &ifst, MutableFst* ofst, unsigned n)
    cdef void ArcMap (Fst &ifst, MutableFst* ofst, ArcMapper mapper)

    cdef void ShortestDistance(Fst &fst, vector[TropicalWeight]* distance, bint reverse)
    cdef void StdArcPushInitial "fst::Push<fst::StdArc, fst::REWEIGHT_TO_INITIAL>" (Fst &ifst,
        MutableFst* ofst, uint32_t ptype)
    cdef void StdArcPushFinal "fst::Push<fst::StdArc, fst::REWEIGHT_TO_FINAL>" (Fst &ifst,
        MutableFst* ofst, uint32_t ptype)
    cdef void RandGen(Fst &ifst, MutableFst* ofst, const RandGenOptions& opts)

    # Destructive operations
    cdef void Closure(MutableFst* ifst, ClosureType type)
    cdef void Invert(MutableFst* ifst)
    cdef void Minimize(MutableFst* fst)
    cdef void Project(MutableFst* fst, ProjectType type)
    cdef void Relabel(MutableFst* fst, 
            vector[pair[int, int]]& ipairs,
            vector[pair[int, int]]& opairs)
    cdef void RmEpsilon(MutableFst* fst)
    cdef void TopSort(MutableFst* fst)

    cdef void ArcSort(MutableFst* fst, ILabelCompare[StdArc]& compare)
    cdef void ArcSort(MutableFst* fst, OLabelCompare[StdArc]& compare)
    cdef void Prune(MutableFst* ifst, TropicalWeight threshold)
    cdef void Connect(MutableFst *fst)
    cdef void StdArcReweight "fst::Reweight<fst::StdArc>" (MutableFst* fst,
        vector[TropicalWeight] potentials, ReweightType rtype)

    # Other
    cdef void Union(MutableFst* ifst1, Fst &ifst2)
    cdef void Concat(MutableFst* ifst1, Fst &ifst2)


    ctypedef Fst* ConstStdVectorFstPtr 'const fst::Fst<fst::StdArc>*'
    cdef void Replace(vector[pair[int, ConstStdVectorFstPtr]] label_fst_pairs, 
             MutableFst *ofst,
             int root,
             bint epsilon_on_replace)
