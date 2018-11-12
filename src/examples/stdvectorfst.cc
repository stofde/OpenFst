

#include <fst/equal.h>
#include <fst/matcher.h>
#include <fst/vector-fst.h>
#include <fst/verify.h>
#include <fst/minimize.h>
#include <fst/properties.h>
#include <fst/script/compile-impl.h>

#include <fst/compact-fst.h>
#include <fst/const-fst.h>
#include <fst/edit-fst.h>
#include <fst/matcher-fst.h>
#include <iostream>
#include <vector>

//typedef fst::VectorFst<fst::StdArc> StdVectorFst;

using fst::SymbolTable;
using fst::StdArc;
using fst::StdVectorFst;

void string_fst(
	const std::vector<string> &symbols, 
	const SymbolTable &symbolTable,
	StdVectorFst *fst) {

	auto q = fst->AddState();
	//symbolTable.Find()
	fst->SetFinal(q, StdVectorFst::Weight::One());
}


int main(int argc, char **argv) {


	// A vector FST is a general mutable FST 
	StdVectorFst fst;

	// Adds state 0 to the initially empty FST and make it the start state. 
	fst.AddState();   // 1st state will be state 0 (returned by AddState) 
	fst.SetStart(0);  // arg is state ID

	// Adds two arcs exiting state 0.
	// Arc constructor args: ilabel, olabel, weight, dest state ID. 
	fst.AddArc(0, fst::StdArc(1, 1, 0.5, 1));  // 1st arg is src state ID 
	fst.AddArc(0, fst::StdArc(2, 2, 1.5, 1));

	// Adds state 1 and its arc.
	fst.AddState();
	fst.AddArc(1, fst::StdArc(3, 3, 2.5, 2));

	// Adds state 2 and set its final weight.
	fst.AddState();
	fst.SetFinal(2, 3.5);  // 1st arg is state ID, 2nd arg weight 

	std::cout << "Is acceptor: " << (fst.Properties(fst::kAcceptor, true) != 0) << std::endl;

	fst::Minimize(&fst);

	auto matcher = fst.InitMatcher(fst::MatchType::MATCH_INPUT);
	//fst.Write("binary.fst");

	return 0;
}
