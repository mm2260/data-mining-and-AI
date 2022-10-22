// @author: Mohammed Mehboob

package edu.rit.resolver;

import edu.rit.resolver.core.Clause;
import edu.rit.resolver.core.Predicate;
import edu.rit.resolver.core.Term;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;

public class Resolver {

    // Knowledge Base:
    public static final HashSet<String> constants = new HashSet<>();
    public static final HashSet<String> variables = new HashSet<>();
    public static final HashSet<String> functions = new HashSet<>();
    public static final HashSet<String> predicates = new HashSet<>();

    /**
     * Perform resolution on a knowledge-base to figure out whether or not it is satisfiable.
     * @param clauses The initial knowledge-base which the resolution algorithm initially starts off with
     * @return Returns whether or not the knowledge-base is satisfiable.
     */
    public static boolean resolution(ArrayList<Clause> clauses) {

        ArrayList<Clause> newClauses;
        ArrayList<Clause> resolvents;

        while(true) {
            newClauses = new ArrayList<>();

            // Traverse through each pair of clauses
            for(int i = 0; i < clauses.size()-1; ++i) {
                for(int j = i+1; j < clauses.size(); ++j) {

                    var ci = clauses.get(i);
                    var cj = clauses.get(j);

                    // get the set of resolvent clauses
                    resolvents = resolve(ci,cj);    // < resolve(Ci, Cj) corresponds to the PL-Resolve function.

                    // check if the resolvents contain the empty clause. If they do, then we return false.
                    for(Clause c : resolvents) {
                        if(c.isEmptyClause()) {
                            return false;
                        }
                    }

                    // new <- resolvents
                    for( Clause resolved : resolvents ) {

                        // Add resolvent to new iff the resolvent clause is
                        // something that doesn't already exist in our set of clauses.
                        boolean exists = false;
                        for( Clause c : clauses ) {
                            if (c.equals(resolved)) {
                                exists = true;
                            }
                        }
                        if(!exists) {
                            newClauses.add(resolved);
                        }
                    }
                }// ends the C-j loop.
            }// ends the C-i loop.

            // the new set being empty implies that new is a subset of clauses, in which case we return true.
            if(newClauses.isEmpty()) {
                return true;
            }

            // update our existing clauses list with all the new clauses we derived
            clauses.addAll(newClauses);

        }//ends while

    }

    /**
     * resolve pairs with complementary literals and return all the possible resolvent clauses.
     * @param c1 The first clause of our Ci-Cj pair.
     * @param c2 The second clause of our Ci-Cj pair.
     * @return All the possible clauses that can be resolved from the Ci and Cj clauses.
     */
    private static ArrayList<Clause> resolve(Clause c1, Clause c2) {

        // Maintain a list of resolvents and substitutions.
        ArrayList<Clause> resolvents = new ArrayList<>();
        // the substitutions hashmap represents mapping we get in our unification process.
        final HashMap<Term, Term> substitutions = new HashMap<>();

        // Iterate pair-wise through the predicate in Ci and Cj:
        for(Predicate p1 : c1.getPredicates()) {
            for(Predicate p2 : c2.getPredicates()) {

                // if the predicates are the same, we can continue.
                if(p1.toString().equals(p2.toString())) {
                    continue;
                }

                // Check if P1 is the complement of P2.
                if( p1.isComplementOf(p2, substitutions) ) {    // isComplementOf will also populate the
                                                                // substitutions HashMap with any unification steps
                                                                // that may have been taken.

                    ArrayList<Predicate> resolvedPredicates = new ArrayList<>();
                    resolvedPredicates.addAll( c1.getPredicates());
                    resolvedPredicates.addAll( c2.getPredicates());
                    resolvedPredicates.remove(p1);
                    resolvedPredicates.remove(p2);

                    ArrayList<Predicate> newPredicates = new ArrayList<>();
                    for(var p : resolvedPredicates) {
                        newPredicates.add( new Predicate(p) );
                    }

                    var newClause = new Clause(newPredicates);
                    resolvents.add( newClause );
                }

            }
        }

        // If there were any substitutions made in the unification process, make sure that the substitutions are
        // applied to all resolvent clauses.
        performSubstitutions(resolvents, substitutions);
        return resolvents;
    }


    /**
     * @param resolvents All the resolvent clauses that need substitutions applied to them.
     * @param substitutions The substitution mappings
     */
    static private void performSubstitutions(ArrayList<Clause> resolvents, HashMap<Term,Term> substitutions) {

        // convert the term:term mapping to string:string mapping.
        HashMap<String, String> mapping = new HashMap<>();
        for(Term t : substitutions.keySet() ) {
            mapping.put( t.toString(), substitutions.get(t).toString() );
        }

        for(Clause c : resolvents ) {
            for(Predicate p : c.getPredicates() ) {
                var arguments = p.getArguments();
                for(int i = 0; i < arguments.size(); ++i ) {
                    var t = arguments.get(i);
                    if(mapping.containsKey(t.toString())){
                        arguments.set( i, new Term(mapping.get(t.toString())) );
                    }
                }// ends the argument iteration.
            }// ends predicate for-loop.
        }//ends clause for-loop.
    }

}