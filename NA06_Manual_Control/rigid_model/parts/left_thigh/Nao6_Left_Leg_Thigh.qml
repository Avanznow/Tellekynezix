import QtQuick
import QtQuick3D

Node {
    id: node

    // Resources
    PrincipledMaterial {
        id: plastico_BLANCO_012_material
        objectName: "PLASTICO_BLANCO.012"
        baseColor: "#ffcccccc"
        indexOfRefraction: 1.4500000476837158
    }
    PrincipledMaterial {
        id: altavoz_025_material
        objectName: "ALTAVOZ.025"
        baseColor: "#ff555555"
        indexOfRefraction: 1.4500000476837158
    }
    PrincipledMaterial {
        id: material_013_material
        objectName: "Material.013"
        baseColor: "#ff141414"
        indexOfRefraction: 1.4500000476837158
    }

    // Nodes:
    Node {
        id: nao6_Left_Leg_Thigh_obj
        objectName: "Nao6_Left_Leg_Thigh.obj"
        Model {
            id: nao6_Left_Leg_Thigh
            objectName: "Nao6_Left_Leg_Thigh"
            source: "meshes/nao6_Left_Leg_Thigh_mesh.mesh"
            materials: [
                plastico_BLANCO_012_material,
                altavoz_025_material,
                material_013_material
            ]
        }
    }

    // Animations:
}
