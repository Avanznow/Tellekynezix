import QtQuick
import QtQuick3D

Node {
    id: node

    // Resources
    PrincipledMaterial {
        id: plastico_BLANCO_020_material
        objectName: "PLASTICO_BLANCO.020"
        baseColor: "#ffcccccc"
        indexOfRefraction: 1.4500000476837158
    }
    PrincipledMaterial {
        id: altavoz_041_material
        objectName: "ALTAVOZ.041"
        baseColor: "#ff555555"
        indexOfRefraction: 1.4500000476837158
    }

    // Nodes:
    Node {
        id: nao6_Right_Leg_Calf_obj
        objectName: "Nao6_Right_Leg_Calf.obj"
        Model {
            id: nao6_Right_Leg_Calf
            objectName: "Nao6_Right_Leg_Calf"
            source: "meshes/nao6_Right_Leg_Calf_mesh.mesh"
            materials: [
                plastico_BLANCO_020_material,
                altavoz_041_material
            ]
        }
    }

    // Animations:
}
