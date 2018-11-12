#!python
#cython: language_level=3

cimport openfst
cimport sym

import subprocess
import random
import re

from libcpp.vector cimport vector
from libcpp.string cimport string
from libcpp.pair cimport pair
from libc.stdint cimport uint64_t
from libc.limits cimport INT_MAX
from util cimport ifstream, ostringstream

EPSILON_ID = 0
EPSILON = u'\u03b5'

cdef bytes as_str(data):
    if isinstance(data, bytes):
        return data
    elif isinstance(data, unicode):
        return data.encode('utf8')
    raise TypeError('Cannot convert {0} to bytestring'.format(type(data)))

def read(filename):
    """read(filename) -> transducer read from the binary file
    Detect arc type (LogArc or TropicalArc) and produce specific transducer."""
    filename = as_str(filename)
    cdef ifstream* stream = new ifstream(filename)
    cdef openfst.FstHeader header
    header.Read(stream[0], filename)
    cdef bytes arc_type = header.ArcType()
    del stream
    if arc_type == b'standard':
        return read_std(filename)
    elif arc_type == b'log':
        # return read_log(filename)
        raise TypeError('log machines currently not supported')
    raise TypeError('cannot read transducer with arcs of type {0}'.format(arc_type))

def read_std(filename):
    """read_std(filename) -> StdVectorFst read from the binary file"""
    cdef StdVectorFst fst = StdVectorFst.__new__(StdVectorFst)
    fst.fst = openfst.StdVectorFstRead(as_str(filename))
    fst._init_tables()
    return fst

#def read_log(filename):
#    """read_log(filename) -> LogVectorFst read from the binary file"""
#    cdef LogVectorFst fst = LogVectorFst.__new__(LogVectorFst)
#    fst.fst = openfst.LogVectorFstRead(as_str(filename))
#    fst._init_tables()
#    return fst

def read_symbols(filename):
    """read_symbols(filename) -> SymbolTable read from the binary file"""
    filename = as_str(filename)
    cdef ifstream* fstream = new ifstream(filename)
    cdef SymbolTable table = SymbolTable.__new__(SymbolTable)
    table.table = sym.SymbolTableRead(fstream[0], filename)
    del fstream
    return table


cdef class SymbolTable:
    cdef sym.SymbolTable* table

    def __init__(self, epsilon=EPSILON):
        """SymbolTable() -> new symbol table with \u03b5 <-> 0
        SymbolTable(epsilon) -> new symbol table with epsilon <-> 0"""
        cdef bytes name = 'SymbolTable<{0}>'.format(id(self)).encode('ascii')
        self.table = new sym.SymbolTable(<string> name)
        assert (self[epsilon] == EPSILON_ID)

    def __dealloc__(self):
        del self.table

    def copy(self):
        """table.copy() -> copy of the symbol table"""
        cdef SymbolTable result = SymbolTable.__new__(SymbolTable)
        result.table = new sym.SymbolTable(self.table[0])
        return result

    def __getitem__(self, sym):
        return self.table.AddSymbol(as_str(sym))

    def __setitem__(self, sym, long key):
        self.table.AddSymbol(as_str(sym), key)

    def write(self, filename):
        """table.write(filename): save the symbol table to filename"""
        self.table.Write(as_str(filename))

    def find(self, key):
        """table.find(int value) -> decoded symbol if any symbol maps to this value
        table.find(str symbol) -> encoded value if this symbol is in the table"""
        if isinstance(key, (int, long)):
            result = self.table.Find(<long> key).decode('utf8')
            if result == u'':
                raise KeyError(key)
            return result
        else:
            key = as_str(key)
            result = self.table.Find(<char*>key)
            if result == -1:
                raise KeyError(key)
            return result

    def __contains__(self, key):
        key = as_str(key)
        return (self.table.Find(<char*>key) != -1)

    def __len__(self):
        return self.table.NumSymbols()

    def items(self):
        """table.items() -> iterator over (symbol, value) pairs"""
        cdef sym.SymbolTableIterator* it = new sym.SymbolTableIterator(self.table[0])
        try:
            while not it.Done():
                yield (it.Symbol().decode('utf8'), it.Value())
                it.Next()
        finally:
            del it

    def __richcmp__(SymbolTable x, SymbolTable y, int op):
        if op == 2: # ==
            return x.table.LabeledCheckSum() == y.table.LabeledCheckSum()
        elif op == 3: # !=
            return not (x == y)
        raise NotImplementedError('comparison not implemented for SymbolTable')

    def __repr__(self):
        return '<SymbolTable of size {0}>'.format(len(self))

    def merge(self, other):
        """table.merge(other): if the two tables are compatible,
        extend this table with the symbol, value pairs of the other table"""
        if other is None or other == self:
            return
        _merge_tables(self, other, self)
        _merge_tables(other, self, self)


def _merge_tables(SymbolTable syms1, SymbolTable syms2, SymbolTable merged):
    """
    Merge tables `syms1` and `syms2` into `merged` if they are compatible.
    Tables are compatible if all common symbol/values map identically.
    """
    for symbol, value in syms1.items():
        try:
            other_symbol = syms2.find(value)
            if other_symbol != symbol:
                raise ValueError('incompatible symbol tables')
        except KeyError:
            pass
        try:
            other_value = syms2.find(symbol)
            if other_value != value:
                raise ValueError('incompatible symbol tables')
        except KeyError:
            pass
        merged[symbol] = value


cdef class _Fst:
    def __init__(self):
        raise NotImplementedError('use StdVectorFst or LogVectorFst to create a transducer')

    def _repr_svg_(self):
        """IPython magic: show SVG reprensentation of the transducer"""
        try:
            process = subprocess.Popen(['dot', '-Tsvg'], stdin=subprocess.PIPE,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except OSError:
            raise Exception('cannot find the dot binary')
        out, err = process.communicate(self.draw())
        if err:
            raise Exception(err)
        return out


cdef class TropicalWeight:
    cdef openfst.TropicalWeight* weight

    ZERO = TropicalWeight(False)
    ONE = TropicalWeight(True)

    def __init__(self, value):
        """TropicalWeight(value) -> tropical weight initialized with the given value"""
        if value is True or value is None:
            self.weight = new openfst.TropicalWeight(openfst.TropicalWeightOne())
        elif value is False:
            self.weight = new openfst.TropicalWeight(openfst.TropicalWeightZero())
        else:
            self.weight = new openfst.TropicalWeight(float(value))

    def __dealloc__(self):
        del self.weight

    def __float__(self):
        return self.weight.Value()

    def __int__(self):
        return int(self.weight.Value())

    def __bool__(self):
        return not (self.weight[0] == openfst.TropicalWeightZero())

    def __richcmp__(TropicalWeight x, TropicalWeight y, int op):
        if op == 2: # ==
            return x.weight[0] == y.weight[0]
        elif op == 3: # !=
            return not (x == y)
        raise NotImplementedError('comparison not implemented for TropicalWeight')

    def approx_equal(self, TropicalWeight other):
        return openfst.ApproxEqual(self.weight[0], other.weight[0])

    def __add__(TropicalWeight x, TropicalWeight y):
        cdef TropicalWeight result = TropicalWeight.__new__(TropicalWeight)
        result.weight = new openfst.TropicalWeight(openfst.Plus(x.weight[0], y.weight[0]))
        return result

    def __mul__(TropicalWeight x, TropicalWeight y):
        cdef TropicalWeight result = TropicalWeight.__new__(TropicalWeight)
        result.weight = new openfst.TropicalWeight(openfst.Times(x.weight[0], y.weight[0]))
        return result

    def __div__(TropicalWeight x, TropicalWeight y):
        cdef TropicalWeight result = TropicalWeight.__new__(TropicalWeight)
        result.weight = new openfst.TropicalWeight(openfst.Divide(x.weight[0], y.weight[0]))
        return result

    def __repr__(self):
        return 'TropicalWeight({0})'.format(float(self))



cdef class StdArc:
    cdef openfst.StdArc* arc
    SEMIRING = TropicalWeight

    def __init__(self):
        """A StdVectorFst arc (with a tropical weight)"""
        raise NotImplementedError('cannot create independent arc')

    property ilabel:
        def __get__(self):
            return self.arc.ilabel

        def __set__(self, int ilabel):
            self.arc.ilabel = ilabel

    property olabel:
        def __get__(self):
            return self.arc.olabel

        def __set__(self, int olabel):
            self.arc.olabel = olabel

    property nextstate:
        def __get__(self):
            return self.arc.nextstate

        def __set__(self, int nextstate):
            self.arc.nextstate = nextstate

    property weight:
        def __get__(self):
            cdef TropicalWeight weight = TropicalWeight.__new__(TropicalWeight)
            weight.weight = new openfst.TropicalWeight(self.arc.weight)
            return weight

        def __set__(self, TropicalWeight weight):
            # TODO create function as_weight(anything) -> Weight
            self.arc.weight = weight.weight[0]

    def __repr__(self):
        return '<StdArc -> {0} | {1}:{2}/{3}>'.format(self.nextstate,
            self.ilabel, self.olabel, self.weight)

cdef class StdState:
    cdef public int stateid
    cdef openfst.StdVectorFst* fst
    SEMIRING = TropicalWeight

    def __init__(self):
        """A StdVectorFst state (with StdArc arcs)"""
        raise NotImplementedError('cannot create independent state')

    def __len__(self):
        return self.fst.NumArcs(self.stateid)

    def __iter__(self):
        cdef openfst.ArcIterator[openfst.StdVectorFst]* it
        it = new openfst.ArcIterator[openfst.StdVectorFst](self.fst[0], self.stateid)
        cdef StdArc arc
        try:
            while not it.Done():
                arc = StdArc.__new__(StdArc)
                arc.arc = <openfst.StdArc*> &it.Value()
                yield arc
                it.Next()
        finally:
            del it

    property arcs:
        """state.arcs: all the arcs starting from this state"""
        def __get__(self):
            return iter(self)

    property final:
        def __get__(self):
            cdef TropicalWeight weight = TropicalWeight.__new__(TropicalWeight)
            weight.weight = new openfst.TropicalWeight(self.fst.Final(self.stateid))
            return weight

        def __set__(self, weight):
            if not isinstance(weight, TropicalWeight):
                weight = TropicalWeight(weight)
            self.fst.SetFinal(self.stateid, (<TropicalWeight> weight).weight[0])

    property initial:
        def __get__(self):
            return self.stateid == self.fst.Start()

        def __set__(self, v):
            if v:
                self.fst.SetStart(self.stateid)
            elif self.stateid == self.fst.Start():
                self.fst.SetStart(-1)

    def __repr__(self):
        return '<StdState #{0} with {1} arcs>'.format(self.stateid, len(self))

cdef class StdVectorFst(_Fst):
    cdef openfst.StdVectorFst* fst
    cdef public SymbolTable isyms, osyms
    SEMIRING = TropicalWeight

    def __init__(self, source=None, isyms=None, osyms=None):
        """StdVectorFst(isyms=None, osyms=None) -> empty finite-state transducer
        StdVectorFst(source) -> copy of the source transducer"""
        if isinstance(source, StdVectorFst):
            self.fst = <openfst.StdVectorFst*> self.fst.Copy()
        else:
            self.fst = new openfst.StdVectorFst()
            # todo: implement LogVectorFst.
            #if isinstance(source, LogVectorFst):
            #    openfst.ArcMap((<LogVectorFst> source).fst[0], self.fst,
            #        openfst.LogToStdWeightConvertMapper())
            #    isyms, osyms = source.isyms, source.osyms
        # Copy symbol tables (of source or given)
        if isyms is not None:
            self.isyms = isyms.copy()
        if osyms is not None:
            self.osyms = (self.isyms if (isyms is osyms) else osyms.copy())

    def __dealloc__(self):
        del self.fst, self.isyms, self.osyms

    def _init_tables(self):
        if self.fst.MutableInputSymbols() != NULL:
            self.isyms = SymbolTable.__new__(SymbolTable)
            self.isyms.table = new sym.SymbolTable(self.fst.MutableInputSymbols()[0])
            self.fst.SetInputSymbols(NULL)
        if self.fst.MutableOutputSymbols() != NULL:
            self.osyms = SymbolTable.__new__(SymbolTable)
            self.osyms.table = new sym.SymbolTable(self.fst.MutableOutputSymbols()[0])
            self.fst.SetOutputSymbols(NULL)
        # reduce memory usage if isyms == osyms
        if self.isyms == self.osyms:
            self.osyms = self.isyms

    def __len__(self):
        return self.fst.NumStates()

    def num_arcs(self):
        """fst.num_arcs() -> total number of arcs in the transducer"""
        return sum(len(state) for state in self)

    def __repr__(self):
        return '<StdVectorFst with {0} states>'.format(len(self))

    def copy(self):
        """fst.copy() -> a copy of the transducer"""
        cdef StdVectorFst result = StdVectorFst.__new__(StdVectorFst)
        if self.isyms is not None:
            result.isyms = self.isyms.copy()
        if self.osyms is not None:
            result.osyms = (result.isyms if (self.isyms is self.osyms)
                else self.osyms.copy())
        result.fst = <openfst.StdVectorFst*> self.fst.Copy()
        return result

    def __getitem__(self, int stateid):
        if not (0 <= stateid < len(self)):
            raise KeyError('state index out of range')
        cdef StdState state = StdState.__new__(StdState)
        state.stateid = stateid
        state.fst = self.fst
        return state

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    property states:
        def __get__(self):
            return iter(self)

    property start:
        def __get__(self):
            return self.fst.Start()
        
        def __set__(self, int start):
            self.fst.SetStart(start)

    def add_arc(self, int source, int dest, int ilabel, int olabel, weight=None):
        """fst.add_arc(int source, int dest, int ilabel, int olabel, weight=None)
        add an arc source->dest labeled with labels ilabel:olabel and weighted with weight"""
        if source > self.fst.NumStates()-1:
            raise ValueError('invalid source state id ({0} > {1})'.format(source,
                self.fst.NumStates()-1))
        if dest > self.fst.NumStates()-1:
            raise ValueError('invalid desination state id ({0} > {1})'.format(dest,
                self.fst.NumStates()-1))
        if not isinstance(weight, TropicalWeight):
            weight = TropicalWeight(weight)
        cdef openfst.StdArc* arc
        arc = new openfst.StdArc(ilabel, olabel, (<TropicalWeight> weight).weight[0], dest)
        self.fst.AddArc(source, arc[0])
        del arc

    def add_state(self):
        """fst.add_state() -> new state"""
        return self.fst.AddState()

    def __richcmp__(StdVectorFst x, StdVectorFst y, int op):
        if op == 2: # ==
            return openfst.Equivalent(x.fst[0], y.fst[0]) # FIXME check deterministic eps-free
        elif op == 3: # !=
            return not (x == y)
        raise NotImplementedError('comparison not implemented for StdVectorFst')

    def write(self, filename, keep_isyms=False, keep_osyms=False):
        """fst.write(filename): write the binary representation of the transducer in filename"""
        if keep_isyms and self.isyms is not None:
            self.fst.SetInputSymbols(self.isyms.table)
        if keep_osyms and self.osyms is not None:
            self.fst.SetOutputSymbols(self.osyms.table)
        result = self.fst.Write(as_str(filename))
        # reset symbols:
        self.fst.SetInputSymbols(NULL)
        self.fst.SetOutputSymbols(NULL)
        return result

    property input_deterministic:
        def __get__(self):
            return bool(self.fst.Properties(openfst.kIDeterministic, True) &
                openfst.kIDeterministic)

    property output_deterministic:
        def __get__(self):
            return bool(self.fst.Properties(openfst.kODeterministic, True) &
                openfst.kODeterministic)

    property acceptor:
        def __get__(self):
            return bool(self.fst.Properties(openfst.kAcceptor, True) &
                openfst.kAcceptor)

    def determinize(self):
        """fst.determinize() -> determinized transducer"""
        cdef StdVectorFst result = StdVectorFst(isyms=self.isyms, osyms=self.osyms)
        openfst.Determinize(self.fst[0], result.fst)
        return result

    def compose(self, StdVectorFst other):
        """fst.compose(StdVectorFst other) -> composed transducer
        Shortcut: fst >> other"""
        if (self.osyms or other.isyms) and (self.osyms != other.isyms):
            raise ValueError('transducer symbol tables are not compatible for composition')
        cdef StdVectorFst result = StdVectorFst(isyms=self.isyms, osyms=other.osyms)
        openfst.Compose(self.fst[0], other.fst[0], result.fst)
        return result

    def __rshift__(StdVectorFst x, StdVectorFst y):
        return x.compose(y)

    def intersect(self, StdVectorFst other):
        """fst.intersect(StdVectorFst other) -> intersection of the two acceptors
        Shortcut: fst & other"""
        if not (self.acceptor and other.acceptor):
            raise ValueError('both transducers need to be acceptors for intersection')
        # TODO check and merge symbol tables (intersection)
        if self.isyms and (self.isyms != other.isyms):
            raise ValueError('transducers must use shared input symbol table')
        if self.osyms and (self.osyms != other.osyms):
            raise ValueError('transducers must use shared output symbol table')
        cdef StdVectorFst result = StdVectorFst(isyms=self.isyms, osyms=self.osyms)
        openfst.Intersect(self.fst[0], other.fst[0], result.fst)
        return result

    def __and__(StdVectorFst x, StdVectorFst y):
        return x.intersect(y)

    def set_union(self, StdVectorFst other):
        """fst.set_union(StdVectorFst other): modify to the union of the two transducers"""
        if self.isyms:
            self.isyms.merge(other.isyms)
        if self.osyms:
            self.osyms.merge(other.osyms)
        openfst.Union(self.fst, other.fst[0])

    def union(self, StdVectorFst other):
        """fst.union(StdVectorFst other) -> union of the two transducers
        Shortcut: fst | other"""
        cdef StdVectorFst result = self.copy()
        result.set_union(other)
        return result

    def __or__(StdVectorFst x, StdVectorFst y):
        return x.union(y)

    def concatenate(self, StdVectorFst other):
        """fst.concatenate(StdVectorFst other): modify to the concatenation
        of the two transducers"""
        if self.isyms:
            self.isyms.merge(other.isyms)
        if self.osyms:
            self.osyms.merge(other.osyms)
        openfst.Concat(self.fst, other.fst[0])

    def concatenation(self, StdVectorFst other):
        """fst.concatenation(StdVectorFst other) -> concatenation of the two transducers
        Shortcut: fst + other"""
        cdef StdVectorFst result = self.copy()
        result.concatenate(other)
        return result

    def __add__(StdVectorFst x, StdVectorFst y):
        return x.concatenation(y)

    def difference(self, StdVectorFst other):
        """fst.difference(StdVectorFst other) -> difference of the two transducers
        Shortcut: fst - other"""
        # TODO merge symbol tables (union)
        if self.isyms and (self.isyms != other.isyms):
            raise ValueError('transducers must use shared input symbol table')
        if self.osyms and (self.osyms != other.osyms):
            raise ValueError('transducers must use shared output symbol table')
        cdef StdVectorFst result = StdVectorFst(isyms=self.isyms, osyms=self.osyms)
        openfst.Difference(self.fst[0], other.fst[0], result.fst)
        return result

    def __sub__(StdVectorFst x, StdVectorFst y):
        return x.difference(y)

    def set_closure(self, plus=False):
        """fst.set_closure(): modify to the Kleene closure of the transducer"""
        openfst.Closure(self.fst, (openfst.CLOSURE_PLUS if plus
            else openfst.CLOSURE_STAR))

    def closure(self):
        """fst.closure() -> Kleene closure of the transducer"""
        cdef StdVectorFst result = self.copy()
        result.set_closure()
        return result

    def closure_plus(self):
        """fst.closure_plus() -> Kleen plus closure (X+ = XX*) of the transducer"""
        cdef StdVectorFst result = self.copy()
        result.set_closure(plus=True)
        return result

    def invert(self):
        """fst.invert(): modify to the inverse of the transducer
        switch input and output labels"""
        openfst.Invert(self.fst)
        self.isyms, self.osyms = self.osyms, self.isyms
    
    def inverse(self):
        """fst.inverse() -> inverse of the transducer"""
        cdef StdVectorFst result = self.copy()
        result.invert()
        return result

    def reverse(self):
        """fst.reverse() -> reversed transducer"""
        cdef StdVectorFst result = StdVectorFst(isyms=self.isyms, osyms=self.osyms)
        openfst.Reverse(self.fst[0], result.fst)
        return result

    def shortest_distance(self, bint reverse=False):
        """fst.shortest_distance(bool reverse=False) -> length of the shortest path"""
        cdef vector[openfst.TropicalWeight] distances
        openfst.ShortestDistance(self.fst[0], &distances, reverse)
        cdef unsigned i
        dist = [TropicalWeight(distances[i].Value()) for i in range(distances.size())]
        return dist

    def shortest_path(self, unsigned n=1):
        """fst.shortest_path(int n=1) -> transducer containing the n shortest paths"""
        if not isinstance(self, StdVectorFst):
            raise TypeError('Weight needs to have the path property and be right distributive')
        cdef StdVectorFst result = StdVectorFst(isyms=self.isyms, osyms=self.osyms)
        openfst.ShortestPath(self.fst[0], result.fst, n)
        return result

    def push(self, final=False, weights=False, labels=False):
        """fst.push(final=False, weights=False, labels=False) -> transducer with
        weights or/and labels pushed to initial (default) or final state"""
        cdef StdVectorFst result = StdVectorFst(isyms=self.isyms, osyms=self.osyms)
        cdef int ptype = 0
        if weights: ptype |= openfst.kPushWeights
        if labels: ptype |= openfst.kPushLabels
        if final:
            openfst.StdArcPushFinal(self.fst[0], result.fst, ptype)
        else:
            openfst.StdArcPushInitial(self.fst[0], result.fst, ptype)
        return result

    def push_weights(self, final=False):
        """fst.push_weights(final=False) -> transducer with weights pushed
        to initial (default) or final state"""
        return self.push(final, weights=True)

    def push_labels(self, final=False):
        """fst.push_labels(final=False) -> transducer with labels pushed
        to initial (default) or final state"""
        return self.push(final, labels=True)

    def reweight(self, potentials, final=False):
        """fst.reweight(potentials, final=False): reweight arcs with given
        potentials in direction of initial (default) or final state"""
        if not len(potentials) == len(self):
            raise ValueError('potential list of invalid length')
        cdef rtype = (openfst.REWEIGHT_TO_FINAL if final else
                openfst.REWEIGHT_TO_INITIAL)
        cdef vector[openfst.TropicalWeight] potentials_vector = vector[openfst.TropicalWeight]()
        for weight in potentials:
            if not isinstance(weight, TropicalWeight):
                weight = TropicalWeight(weight)
            potentials_vector.push_back((<TropicalWeight> weight).weight[0])
        openfst.StdArcReweight(self.fst, potentials_vector, rtype)

    def minimize(self):
        """fst.minimize(): minimize the transducer"""
        if not self.input_deterministic:
            raise ValueError('transducer is not input deterministic')
        openfst.Minimize(self.fst)

    def arc_sort_input(self):
        """fst.arc_sort_input(): sort the input arcs of the transducer"""
        cdef openfst.ILabelCompare[openfst.StdArc] icomp
        openfst.ArcSort(self.fst, icomp)

    def arc_sort_output(self):
        """fst.arc_sort_output(): sort the output arcs of the transducer"""
        cdef openfst.OLabelCompare[openfst.StdArc] ocomp
        openfst.ArcSort(self.fst, ocomp)

    def top_sort(self):
        """fst.top_sort(): topologically sort the nodes of the transducer"""
        openfst.TopSort(self.fst)

    def project_input(self):
        """fst.project_input(): project the transducer on the input side"""
        openfst.Project(self.fst, openfst.PROJECT_INPUT)
        self.osyms = self.isyms

    def project_output(self):
        """fst.project_output(): project the transducer on the output side"""
        openfst.Project(self.fst, openfst.PROJECT_OUTPUT)
        self.isyms = self.osyms

    def remove_epsilon(self):
        """fst.remove_epsilon(): remove the epsilon transitions from the transducer"""
        openfst.RmEpsilon(self.fst)

    def _tosym(self, label, io):
        # If integer label, return integer
        if isinstance(label, int):
            return label
        # Otherwise, try to convert using symbol tables
        if io and self.isyms is not None:
            return self.isyms[label]
        elif not io and self.osyms is not None:
            return self.osyms[label]
        raise TypeError('Cannot convert label "{0}" to symbol'.format(label))

    def relabel(self, imap={}, omap={}):
        """fst.relabel(imap={}, omap={}): relabel the symbols on the arcs of the transducer
        imap/omap: (int -> int) or (str -> str) symbol mappings"""
        cdef vector[pair[int, int]] ip
        cdef vector[pair[int, int]] op
        for old, new in imap.items():
            ip.push_back(pair[int, int](self._tosym(old, True), self._tosym(new, True)))
        for old, new in omap.items():
            op.push_back(pair[int, int](self._tosym(old, False), self._tosym(new, False)))
        openfst.Relabel(self.fst, ip, op)

    def prune(self, threshold):
        """fst.prune(threshold): prune the transducer"""
        if not isinstance(threshold, TropicalWeight):
            threshold = TropicalWeight(threshold)
        openfst.Prune(self.fst, (<TropicalWeight> threshold).weight[0])

    def connect(self):
        """fst.connect(): removes states and arcs that are not on successful paths."""
        openfst.Connect(self.fst)

    def plus_map(self, value):
        """fst.plus_map(value) -> transducer with weights equal to the original weights
        plus the given value"""
        cdef StdVectorFst result = StdVectorFst(isyms=self.isyms, osyms=self.osyms)
        if not isinstance(value, TropicalWeight):
            value = TropicalWeight(value)
        openfst.ArcMap(self.fst[0], result.fst,
            openfst.PlusStdArcMapper((<TropicalWeight> value).weight[0]))
        return result

    def times_map(self, value):
        """fst.times_map(value) -> transducer with weights equal to the original weights
        times the given value"""
        cdef StdVectorFst result = StdVectorFst(isyms=self.isyms, osyms=self.osyms)
        if not isinstance(value, TropicalWeight):
            value = TropicalWeight(value)
        openfst.ArcMap(self.fst[0], result.fst,
            openfst.TimesStdArcMapper((<TropicalWeight> value).weight[0]))
        return result

    def remove_weights(self):
        """fst.remove_weights() -> transducer with weights removed"""
        cdef StdVectorFst result = StdVectorFst(isyms=self.isyms, osyms=self.osyms)
        openfst.ArcMap(self.fst[0], result.fst, openfst.RmTropicalWeightMapper())
        return result

    def invert_weights(self):
        """fst.invert_weights() -> transducer with inverted weights"""
        cdef StdVectorFst result = StdVectorFst(isyms=self.isyms, osyms=self.osyms)
        openfst.ArcMap(self.fst[0], result.fst, openfst.InvertTropicalWeightMapper())
        return result

    def replace(self, label_fst_map, epsilon=False):
        """fst.replace(label_fst_map, epsilon=False) -> transducer with non-terminals replaced
        label_fst_map: non-terminals (str) -> fst map
        epsilon: replace input label by epsilon?"""
        assert self.osyms # used to encode labels
        cdef StdVectorFst result = StdVectorFst(isyms=self.isyms, osyms=self.osyms)
        cdef vector[pair[int, openfst.ConstStdVectorFstPtr]] label_fst_pairs
        cdef StdVectorFst fst
        label_fst_map['__ROOT__'] = self
        for label, fst in label_fst_map.items():
            assert (not fst.osyms or fst.osyms == self.osyms) # output symbols must match
            label_id = self.osyms[label]
            label_fst_pairs.push_back(pair[int, openfst.ConstStdVectorFstPtr](label_id, fst.fst))
        openfst.Replace(label_fst_pairs, result.fst, self.osyms['__ROOT__'], epsilon)
        return result

    def random_generate(self, n_path=1, max_len=None, uniform=True, weighted=False):
        if uniform:
            return self.uniform_generate(n_path, max_len, weighted)
        else:
            return self.logprob_generate(n_path, max_len, weighted)

    def logprob_generate(self, n_path=1, max_len=None, weighted=False):
        """fst.logprob_generate(n_path=1) -> n_path random paths
        sampled according to weights assumed to encode log probabilities"""
        cdef StdVectorFst result = StdVectorFst(isyms=self.isyms, osyms=self.osyms)
        cdef int seed = random.randint(0, INT_MAX)
        cdef openfst.LogProbStdArcSelector* selector = new openfst.LogProbStdArcSelector(seed)
        cdef int maxlen = (INT_MAX if max_len is None else max_len)
        cdef openfst.LogProbStdArcRandGenOptions* options = new openfst.LogProbStdArcRandGenOptions(selector[0], maxlen, n_path, weighted)
        openfst.RandGen(self.fst[0], result.fst, options[0])
        del options, selector
        return result

    def uniform_generate(self, n_path=1, max_len=None, weighted=False):
        """fst.uniform_generate(n_path=1) -> n_path random paths sampled uniformly"""
        cdef StdVectorFst result = StdVectorFst(isyms=self.isyms, osyms=self.osyms)
        cdef int seed = random.randint(0, INT_MAX)
        cdef openfst.UniformStdArcSelector* selector = new openfst.UniformStdArcSelector(seed)
        cdef int maxlen = (INT_MAX if max_len is None else max_len)
        cdef openfst.UniformStdArcRandGenOptions* options = new openfst.UniformStdArcRandGenOptions(selector[0], maxlen, n_path, weighted)
        openfst.RandGen(self.fst[0], result.fst, options[0])
        del options, selector
        return result

    def _visit(self, int stateid, prefix=()):
        """fst._visit(stateid, prefix): depth-first search"""
        if self[stateid].final:
            yield prefix
        for arc in self[stateid]:
            for path in self._visit(arc.nextstate, prefix+(arc,)):
                yield path

    def paths(self):
        """fst.paths() -> iterator over all the paths in the transducer"""
        return self._visit(self.start)
