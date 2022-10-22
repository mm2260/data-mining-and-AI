// @author: Mohammed Mehboob

package edu.rit.resolver.core;

import edu.rit.resolver.Resolver;

public class Term {

    String name;
    Term argument = null;

    public Term(String name) {
        if(name.contains("(") || name.contains(")")) {
            var split = name.split("[()]");
            this.name = split[0];
            this.argument = new Term(split[1]);
        } else {
            this.name = name;
        }
    }

    public Term(Term t2) {
        this.name = t2.name;
        this.argument = t2.argument==null? null : new Term(t2.argument);
    }

    boolean isVariable() {
        return Resolver.variables.contains(this.name);
    }

    boolean isNotFunction() {
        return !Resolver.functions.contains(this.name);
    }

    @Override
    public String toString() {
        return this.isNotFunction() ? this.name : String.format("%s(%s)",this.name, this.argument);
    }
}
