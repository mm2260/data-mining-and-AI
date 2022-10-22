// @author: Mohammed Mehboob

package edu.rit.resolver.core;

import java.util.ArrayList;
import java.util.HashMap;

public class Predicate {

    private final String name;
    private final boolean negated;
    private final ArrayList<Term> arguments;

    public Predicate(String name, boolean negated, ArrayList<Term> arguments) {
        this.name = name;
        this.negated = negated;
        this.arguments = arguments;
    }

    public Predicate(Predicate p2) {
        this.name = p2.name;
        this.negated = p2.negated;
        this.arguments = new ArrayList<>();
        for(Term arg : p2.arguments ) {
            this.arguments.add( new Term(arg) );
        }
    }

    public ArrayList<Term> getArguments() {
        return arguments;
    }

    public boolean isComplementOf(Predicate p2, HashMap<Term, Term> substitutions) {

        //(1) check if predicate name is the same, and the negation signs are opposite.
        if( this.name.equals(p2.name) && this.negated != p2.negated ) {

            //(2) traverse argument by argument
            for(int i = 0; i < arguments.size(); ++i) {

                var thisTerm = this.arguments.get(i);
                var otherTerm = p2.arguments.get(i);

                //(2a) check if arguments are equal.
                if(!thisTerm.toString().equals(otherTerm.toString())) {

                    //(2b) arguments are not equal:

                    //(3) is one of the arguments a variable?
                    if( thisTerm.isVariable() || otherTerm.isVariable() ) {

                        if(thisTerm.isVariable()) {
                            //(3a) one is a variable. Is the other a function?  << this.arg[i] is the variable.
                            if (unify(substitutions, thisTerm, otherTerm)) return false;
                        } else {
                            //(3a) one is a variable. Is the other a function?  << p2.arg[i] is the variable.
                            if (unify(substitutions, otherTerm, thisTerm)) return false;
                        }
                    } else {
                        // there are no variables. Predicates can't be unified, and the arguments are unequal.
                        // therefore the predicates cannot be complementary.
                        return false;
                    }
                }
            }

            // none of the arguments caused an issue.
            // assuming all arguments were the same, or were unified.
            return true;

         } else {
            // Predicates are unequal and can't be compared.
            return false;
        }

    }

    private boolean unify(HashMap<Term, Term> substitutions, Term thisTerm, Term otherTerm) {
        if(otherTerm.isNotFunction()) {
            //(3b) other is not a function --> replace.
            if(substitutions.containsKey(thisTerm)&&substitutions.get(thisTerm)!=otherTerm){
                return true;
            } else {
                substitutions.put(thisTerm, new Term(otherTerm));
            }
        } else {

            if(!otherTerm.isVariable()) {
                //(4a) other is a function
                //(5) is the function's argument the variable ?
                if (!otherTerm.argument.toString().equals(thisTerm.toString())) {
                    //no:
                    if (substitutions.containsKey(thisTerm) && substitutions.get(thisTerm) != otherTerm) {
                        return true;
                    } else {
                        substitutions.put(thisTerm, new Term(otherTerm));
                    }

                } else {
                    // arguments don't unify: therefore, predicates cannot be complementary.
                    return true;
                }
            } else {
                // both a variables, simply substitute.
                if (substitutions.containsKey(thisTerm) && substitutions.get(thisTerm) != otherTerm) {
                    return true;
                } else {
                    substitutions.put(thisTerm, new Term(otherTerm));
                }
            }
        }
        return false;
    }

    @Override
    public String toString() {
        String argumentStr;

        if(arguments.isEmpty()) {
            argumentStr = "";
        } else {
            argumentStr = "(" +
                    arguments.stream().collect(StringBuilder::new, StringBuilder::append, (x,y)-> x.append(",").append(y)) +
                    ")";
        }

        return (this.negated?"!":"") + this.name + argumentStr;
    }

}